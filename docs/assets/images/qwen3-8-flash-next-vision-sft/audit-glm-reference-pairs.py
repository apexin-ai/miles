"""CPU-only audit of the immutable GLM paired answers and Qwen scene join."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

P = Path('/workspace/jlw/miles-q38-vision-sft-20261030')
ROOT = P / 'data/reference-gained-questions-v1'
SOURCE = ROOT / 'glm/recorded'
OUT = ROOT / 'analysis'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rs, prefix='glm'):
    before = prefix + '_baseline_answer'
    after = prefix + '_trained_answer'
    common = [r for r in rs if r[before] is not None and r[after] is not None]
    gain = [r for r in rs if r[before] != r['answer'] == r[after]]
    loss = [r for r in rs if r[before] == r['answer'] != r[after]]
    b = sum(r[before] == r['answer'] for r in rs)
    a = sum(r[after] == r['answer'] for r in rs)
    cb = sum(r[before] == r['answer'] for r in common)
    ca = sum(r[after] == r['answer'] for r in common)
    return dict(n=len(rs), baseline_correct=b, trained_correct=a, net=a-b,
                wrong_to_right=len(gain), right_to_wrong=len(loss),
                baseline_invalid=sum(r[before] is None for r in rs),
                trained_invalid=sum(r[after] is None for r in rs),
                both_valid_n=len(common), both_valid_baseline_correct=cb,
                both_valid_trained_correct=ca, both_valid_net=ca-cb,
                invalid_to_correct=sum(r[before] is None for r in gain),
                correct_to_invalid=sum(r[after] is None for r in loss),
                gains_from_zero=sum(r[before] == 0 for r in gain),
                gain_ids=[r['id'] for r in gain], loss_ids=[r['id'] for r in loss])


def group(rs, key, prefix='glm'):
    return {str(k): summarize([r for r in rs if r[key] == k], prefix)
            for k in sorted({r[key] for r in rs})}


def main():
    assert not OUT.exists(), 'create-only analysis: inspect the existing receipt'
    result = json.loads((SOURCE/'result.json').read_text())
    provenance = result['provenance']
    assert sha(SOURCE/'paired_answers.jsonl') == provenance['paired_answers_sha256']
    scenes = []
    for name, digest in provenance['scene_files_sha256'].items():
        path = SOURCE/name
        assert sha(path) == digest
        scenes += [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
    scene = {r['id']: r for r in scenes if r['dataset_role'] == 'validation'}
    pairs = rows(SOURCE/'paired_answers.jsonl')
    data_file = P/'data/geometry-v1/validation.jsonl'
    assert sha(data_file) == '6f9b21c2cefe866a6b259c11ca9a0969b190494772cf46ba35ec9f025739376e'
    data = {r['id']: r for r in rows(data_file)}
    eval_dir = P/'data/eval-vision32-prompt-v2-validation256-all-a01'
    qfiles = [eval_dir/f'update-{u:06d}-validation.jsonl' for u in (0,32)]
    qwen = [{r['id']: r for r in rows(path)} for path in qfiles]
    assert len(pairs) == len({r['id'] for r in pairs}) == 256
    assert set(scene) == set(data) == set(qwen[0]) == set(qwen[1]) == {r['id'] for r in pairs}
    joined = []
    for pair in pairs:
        sid = pair['id']
        s, d, qb, qt = scene[sid], data[sid], qwen[0][sid], qwen[1][sid]
        answer = int(pair['answer'])
        assert answer == int(s['answer']) == int(d['answer']) == int(qb['answer']) == int(qt['answer'])
        assert s['image_pixel_sha256'] == d['image_pixel_sha256'] == qb['image_pixel_sha256'] == qt['image_pixel_sha256']
        assert s['structure_sha256'] == d['structure_sha256']
        assert s['sample_seed'] == d['sample_seed']
        question = d['messages'][0]['content']
        assert s['question'] == question == qb['question'] == qt['question']
        assert qb['control'] == qt['control'] == 'image'
        assert qb['input_ids_sha256'] == qt['input_ids_sha256']
        meta = s['generator_metadata']
        v = meta['variant']
        assert pair['sheet_kind'] == v['target_kind']
        b, a = pair['baseline_answer'], pair['trained_answer']
        transition = ('wrong_to_right' if b != answer == a else
                      'right_to_wrong' if b == answer != a else
                      'both_correct' if b == answer == a else 'both_wrong')
        joined.append(dict(id=sid, answer=answer, sheet_kind=v['target_kind'],
                           object_kind=v['small_shape_kind'], target_color=v['target_color'],
                           n_sheets=v['n_sheets'], target_is_top=v['target_is_top'],
                           target_sheet_index=v['target_sheet_index'], question=question,
                           image_pixel_sha256=s['image_pixel_sha256'], image_path=d['images'][0],
                           glm_baseline_answer=b, glm_trained_answer=a,
                           qwen_baseline_answer=qb['extracted_answer'],
                           qwen_trained_answer=qt['extracted_answer'],
                           glm_transition=transition,
                           both_glm_valid=b is not None and a is not None))
    total = summarize(joined)
    expected = result['metrics']
    for k in ['n','baseline_correct','trained_correct','wrong_to_right','right_to_wrong',
              'baseline_invalid','trained_invalid','both_valid_n',
              'both_valid_baseline_correct','both_valid_trained_correct']:
        assert total[k] == expected[k], (k,total[k],expected[k])
    assert total['both_valid_net'] == expected['net_correct_change_both_valid']
    gains = [r for r in joined if r['glm_transition'] == 'wrong_to_right']
    # Examples are the lowest scene ID in each declared category, not hand picked.
    predicates = {
        'rectangle_count_at_least_3_both_valid': lambda r:r['sheet_kind']=='rectangle' and r['answer']>=3 and r['both_glm_valid'],
        'circle_count_at_least_2_both_valid': lambda r:r['sheet_kind']=='circle' and r['answer']>=2 and r['both_glm_valid'],
        'triangle_count_at_least_2_both_valid': lambda r:r['sheet_kind']=='triangle' and r['answer']>=2 and r['both_glm_valid'],
        'count_1_both_valid': lambda r:r['answer']==1 and r['both_glm_valid'],
        'validity_recovery': lambda r:r['glm_baseline_answer'] is None,
    }
    examples = {name:min((r for r in gains if fn(r)),key=lambda r:r['id'])
                for name,fn in predicates.items()}
    summary = dict(schema='reference-gained-questions-v1', complete=True,
                   source_commit='74808f6e24254e1f647d78d8a98d714ab9daacb0',
                   source_pair_sha256=sha(SOURCE/'paired_answers.jsonl'),
                   source_result_sha256=sha(SOURCE/'result.json'),
                   validation_data_sha256=sha(data_file),
                   qwen_predictions_sha256={p.name:sha(p) for p in qfiles},
                   exact_scene_join=dict(n=256,ids=True,questions=True,labels=True,pixels=True,structure=True,seeds=True),
                   glm=total, qwen_prompt_v2=summarize(joined,'qwen'),
                   by_sheet_kind=group(joined,'sheet_kind'),
                   by_answer=group(joined,'answer'), by_object_kind=group(joined,'object_kind'),
                   by_n_sheets=group(joined,'n_sheets'),
                   qwen_on_glm_gain_subset=summarize(gains,'qwen'),
                   examples=examples,
                   limitations=[
                       'GLM extracted-answer pairs are public; raw reasoning text is not supplied.',
                       'These are observed answer transitions, not proof of a visual reasoning mechanism.',
                       'GLM starts from a previously language-updated checkpoint; Qwen starts from the native base.',
                       'GLM first-final-line and Qwen last-final-line scorers differ; no raw GLM rescore is possible.',
                       'DeepSeek per-question raw artifacts are not present in the public source; no IDs are inferred.',
                       'No additional training, validation selection, relabeling or filtering is performed.',
                   ])
    OUT.mkdir()
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'paired-questions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in joined))
    with (OUT/'transitions.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(r for r in joined if r['glm_transition'] in ['wrong_to_right','right_to_wrong'])
    print(json.dumps({k:v for k,v in summary.items() if k in ['complete','exact_scene_join','glm','qwen_prompt_v2','examples']},indent=2))


if __name__ == '__main__':
    main()
