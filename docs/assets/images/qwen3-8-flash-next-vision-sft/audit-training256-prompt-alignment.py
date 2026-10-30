"""Qualify exactly 256 paired training predictions, preserving interrupted provenance."""
from pathlib import Path
import hashlib
import json
import statistics
import sys
import tarfile

P = Path('/workspace/jlw/miles-q38-vision-sft-20261030')
S = P/'workdir-train-prompt-v2'
B = P/'data/eval-training256-baseline-prefix-v1'
E = P/'data/eval-training256-prompt-v2-a01'
sys.path.insert(0,str(S))
from examples.qwen3_8_next_vision_sft.score import PROTOCOL, answer_of, paired_answers, metrics_from_answers, read_rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dest = P/'logs/training256-prompt-v2-qualification-v1.json'
    assert not dest.exists()
    assert (P/'logs/training256-prompt-v2-a01.exit').read_text().strip() == '0'
    pf = P/'logs/training256-prompt-v2-a01-protocol.json'
    protocol = json.loads(pf.read_text())
    assert protocol['n'] == 256 and protocol['checkpoint_updates'] == [0,32]
    assert protocol['not_held_out_accuracy'] and not protocol['gold_target_in_prompt']
    assert protocol['user_question_only'] and protocol['baseline_reused_from_interrupted_larger_run']
    parent = json.loads((B/'parent-evaluation.json').read_text())
    m = json.loads((E/'evaluation.json').read_text())
    assert not parent['complete'] and m['complete']
    assert m['protocol'] == parent['protocol'] == PROTOCOL
    expected = dict(splits=['distillation'],controls=['image'],tp_size=8,batch_size=256,
                    max_new_tokens=16384,context_length=32768,enable_thinking=True,
                    reasoning_effort='low',final_answer_format='integer-only',
                    empty_thinking_prefill=False,enable_cuda_graph=True,
                    constrain_integer_answer=False,mem_fraction_static=0.6)
    for manifest in [parent,m]:
        for k,v in expected.items():
            assert manifest['args'][k] == v,(k,manifest['args'][k],v)
    assert parent['args']['limit'] == 0 and not parent['args']['skip_baseline']
    assert m['args']['limit'] == 256 and m['args']['skip_baseline']
    for key in ['model_files_sha256','data_files_sha256','evaluator_sha256','packages','protocol','scorer_sha256']:
        assert parent[key] == m[key],key
    assert m['evaluator_sha256'] == protocol['evaluator_sha256'] == 'e839b888d27c7c9c482d9c812071776bd68a75a84f6dd879a08442da853ef163'
    assert m['packages'] == {'torch':'2.13.0','transformers':'5.12.1','sglang-kernel':'0.4.6.post1','flashinfer-python':'0.6.17'}
    data = P/'data/geometry-v1'
    for name,digest in protocol['data_files_sha256'].items():
        assert sha(data/name) == digest
    source = read_rows(data/'distillation.jsonl')[:256]
    assert len(source) == len({r['id'] for r in source}) == 256
    assert [r['id'] for r in source] == protocol['selected_ids']
    assert all(len(r['messages']) == 1 and r['messages'][0]['role'] == 'user' for r in source)
    files = [B/'update-000000-distillation.jsonl',E/'update-000032-distillation.jsonl']
    assert sha(files[0]) == protocol['baseline_prefix_sha256']
    old = P/'data/eval-training1024-prompt-v2-a01/update-000000-distillation.jsonl'
    assert sha(old) == protocol['parent_raw_sha256']
    assert files[0].read_bytes() == b''.join(old.read_bytes().splitlines(keepends=True)[:256])
    predictions = [read_rows(f) for f in files]
    assert all(len(rows) == 256 for rows in predictions)
    for update,rows in zip([0,32],predictions,strict=True):
        for r,s in zip(rows,source,strict=True):
            assert r['id'] == s['id'] == r['image_id']
            assert r['update'] == update and r['control'] == 'image'
            assert r['image_pixel_sha256'] == s['image_pixel_sha256']
            assert r['assistant_prefill'] == '<think>\n'
            assert r['prediction'] == r['assistant_prefill'] + r['generated_text']
            assert r['extracted_answer'] == answer_of(r['prediction'])
    for a,b in zip(*predictions,strict=True):
        assert a['prompt_tokens'] == b['prompt_tokens']
        assert a['input_ids_sha256'] == b['input_ids_sha256']
    pairs = paired_answers(source,*predictions)
    metrics = metrics_from_answers(pairs)
    assert [r['update'] for r in m['results']] == [32]
    assert m['results'][0]['metrics']['distillation']['image'] == dict(
        n=256,correct=metrics['trained_correct'],invalid=metrics['trained_invalid'],
        accuracy=metrics['trained_correct']/256)
    training = json.loads((P/'logs/train-vision32-prompt-v2-qualification.json').read_text())
    assert training['complete'] and training['updates'] == 32 and training['language_frozen']
    assert sha(P/'logs/train-vision32-prompt-v2-qualification.json') == protocol['training_qualification_sha256']
    host = Path('/workspace/jlw/b300-native/q38-train256-v1-a01')
    receipt = json.loads((host/'execution-receipt.json').read_text())
    assert receipt['state'] == 'completed' and receipt['exit_code'] == 0
    parent_receipt = json.loads((B/'parent-execution-receipt.json').read_text())
    assert parent_receipt['state'] == 'interrupted' and parent_receipt['exit_code'] == 143
    lengths,invalid = {},{}
    for u,rows in zip([0,32],predictions,strict=True):
        lengths[str(u)] = dict(mean_completion_tokens=statistics.mean(r['completion_tokens'] for r in rows),
                               max_completion_tokens=max(r['completion_tokens'] for r in rows),
                               at_token_limit=sum(r['completion_tokens'] >= 16384 for r in rows))
        bad = [r for r in rows if answer_of(r['prediction']) is None]
        invalid[str(u)] = dict(total=len(bad),unfinished_thinking=sum('</think>' not in r['prediction'] for r in bad),
                               ids=[r['id'] for r in bad])
    out = P/'logs/training256-prompt-v2-results-v1'
    out.mkdir()
    (out/'paired-answers.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
    q = dict(schema='training256-prompt-v2-qualification-v1',complete=True,n=256,
             training_set=True,not_held_out_accuracy=True,not_teacher_forced=True,
             selection=protocol['selection'],same_questions_labels_pixels_expanded_ids=True,
             gold_targets_not_in_prompt=True,thinking_enabled=True,scorer=PROTOCOL,
             language_frozen=True,metrics=metrics,completion_lengths=lengths,invalid_breakdown=invalid,
             trained_seconds=m['seconds'],protocol_sha256=sha(pf),
             baseline_parent_was_interrupted=True,baseline_parent_receipt=parent_receipt,
             execution_receipt=receipt,
             files_sha256={str(f.relative_to(P)):sha(f) for f in files+[E/'evaluation.json',B/'parent-evaluation.json']},
             qualifier_sha256=sha(Path(__file__)))
    dest.write_text(json.dumps(q,indent=2)+'\n')
    archive = P/'logs/training256-prompt-v2-evidence-v1.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tar:
        for f in files+[E/'evaluation.json',B/'parent-evaluation.json',B/'parent-execution-receipt.json',pf,dest,
                        P/'logs/training256-prompt-v2-a01.exit',out/'paired-answers.jsonl']:
            tar.add(f,arcname=str(f.relative_to(P)))
        for name in ['eval-training256-prompt-v2-v1.py','host-runtime-exec-training256-v1.py','qualify-training256-prompt-v2-v1.py']:
            tar.add(P/'scripts'/name,arcname='scripts/'+name)
        for name in ['execution-receipt.json','launch-intent.json','deployment.json','scripts/task.sh','logs/task.log']:
            f = host/name
            if f.is_file():tar.add(f,arcname='host-task/'+name)
    print(json.dumps({k:v for k,v in q.items() if k not in ['metrics','files_sha256']},indent=2))
    print(json.dumps({'metrics':{k:v for k,v in metrics.items() if not isinstance(v,dict)}}))
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)}))


if __name__ == '__main__':
    main()
