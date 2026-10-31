#!/bin/bash
# Move apex/main to a validated candidate so that the tree of apex/main equals the tree of the candidate
# (BRANCH-PR plan section 5.3.2). Dry run by default; never force-pushes.
#
#   docs/apex/promote-apex-main.sh --candidate <C> --gate <acceptance record> [--upstream <U>]
#                                  [--coverage-checked] [--remote origin] [--branch apex/main] [--push]
#
# The acceptance record is a separate file written by whoever accepts the gate (not a GPU log). Lines are
# "key: value"; blank lines and lines starting with "#" are ignored; only these keys are allowed:
#   candidate: <full 40-hex SHA>      exactly one, must equal C
#   status: PASSED                    exactly one; any other value (FAILED, NOT_RUN, PENDING, ...) is refused
#   gate: <name>  accepted_by: <who>  date: <UTC>  evidence: <path>   optional, at most one each
# Any other key, a repeated key, or a missing candidate / status line makes the record invalid.
#
# Cases, decided from the remote branch A (git ls-remote):
#   1. A does not exist                         -> apex/main = C (first creation).
#   2. A == C, or C is an ancestor of A and
#      tree(A) == tree(C)                       -> already adopted; nothing to do (no new commit, no push).
#   3. C is an ancestor of A, trees differ      -> refused: apex/main already has changes on top of C.
#   4. A is an ancestor of C                    -> fast-forward apex/main to C.
#   5. otherwise (new upstream, or same tree with unrelated history)
#                                               -> M = commit-tree(C^{tree}, parents A and C); needs --coverage-checked,
#                                                  i.e. every local change of A is in C or was dropped with a written
#                                                  reason (docs/apex/patches.md).
# Asserted before any push: tree(new) == tree(C); A (if any), C and U (if given) are ancestors of the new head;
# the remote branch is still A right before the push.
set -euo pipefail

remote=origin branch=apex/main candidate= gate= upstream= coverage=0 push=0
while [ $# -gt 0 ]; do
  case "$1" in
    --candidate) candidate=$2; shift 2 ;;
    --gate) gate=$2; shift 2 ;;
    --upstream) upstream=$2; shift 2 ;;
    --coverage-checked) coverage=1; shift ;;
    --remote) remote=$2; shift 2 ;;
    --branch) branch=$2; shift 2 ;;
    --push) push=1; shift ;;
    -h | --help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { echo "ABORT: $*" >&2; exit 1; }
[ -n "$candidate" ] || die "--candidate is required"
[ -n "$gate" ] || die "--gate <acceptance record> is required"
[ -f "$gate" ] || die "acceptance record not found: $gate"

# ---- the acceptance record: strict key: value lines
# one variable per key (bash 3.2 on macOS has no associative arrays)
seen_candidate= seen_status= seen_gate= seen_accepted_by= seen_date= seen_evidence=
rec_candidate= rec_status=
while IFS= read -r line || [ -n "$line" ]; do
  line=${line%$'\r'}
  [[ -z "${line//[[:space:]]/}" || "$line" =~ ^[[:space:]]*# ]] && continue
  [[ "$line" =~ ^([a-z_]+):[[:space:]]*(.*[^[:space:]])[[:space:]]*$ ]] || die "acceptance record: malformed line: $line"
  key=${BASH_REMATCH[1]} value=${BASH_REMATCH[2]}
  case "$key" in candidate | status | gate | accepted_by | date | evidence) ;; *) die "acceptance record: unknown key '$key'" ;; esac
  seen_var=seen_$key
  [ -z "${!seen_var}" ] || die "acceptance record: key '$key' appears more than once"
  printf -v "$seen_var" 1
  case "$key" in
    candidate) rec_candidate=$value ;;
    status) rec_status=$value ;;
  esac
done < "$gate"
[ -n "$rec_candidate" ] || die "acceptance record: no candidate line"
[ -n "$rec_status" ] || die "acceptance record: no status line"
[[ "$rec_candidate" =~ ^[0-9a-f]{40}$ ]] || die "acceptance record: candidate is not a full 40-hex SHA: $rec_candidate"
[ "$rec_status" = PASSED ] || die "acceptance record: status is '$rec_status', not PASSED"

git fetch -q "$remote"
C=$(git rev-parse --verify "$candidate^{commit}") || die "candidate is not a commit: $candidate"
[ "$rec_candidate" = "$C" ] || die "acceptance record is for $rec_candidate, not for the candidate $C"
U=
if [ -n "$upstream" ]; then
  U=$(git rev-parse --verify "$upstream^{commit}") || die "upstream is not a commit: $upstream"
  git merge-base --is-ancestor "$U" "$C" || die "upstream $U is not an ancestor of the candidate $C"
fi
tree_of() { git rev-parse "$1^{tree}"; }

A=$(git ls-remote "$remote" "refs/heads/$branch" | cut -f1)
if [ -n "$A" ]; then
  git cat-file -e "$A^{commit}" 2>/dev/null || die "remote $branch = $A is not in the local clone (fetch it)"
fi
if [ -z "$A" ]; then
  case_name="first creation"
  NEW=$C
elif [ "$A" = "$C" ] || { git merge-base --is-ancestor "$C" "$A" && [ "$(tree_of "$A")" = "$(tree_of "$C")" ]; }; then
  echo "$branch ($A) has already adopted $C (C is an ancestor and the trees are equal); nothing to do"
  exit 0
elif git merge-base --is-ancestor "$C" "$A"; then
  die "$branch ($A) already contains $C and has changes on top of it; promoting $C would go back"
elif git merge-base --is-ancestor "$A" "$C"; then
  case_name="fast-forward"
  NEW=$C
else
  case_name="adopt candidate tree on top of $A"
  [ "$coverage" = 1 ] || die "$branch ($A) is not an ancestor of $C: check that every local change of $A is in $C or was dropped with a reason (docs/apex/patches.md), then pass --coverage-checked"
  NEW=$(git commit-tree "$C^{tree}" -p "$A" -p "$C" -m "apex/main: adopt candidate ${C:0:12}

The tree is exactly the tree of ${C:0:12}. Acceptance record: $(basename "$gate") (status PASSED,
candidate $C${seen_gate:+, gate recorded}).
Parents: the previous apex/main ${A:0:12} and the candidate.${U:+ Upstream base: ${U:0:12}.}")
fi

# ---- assertions on the new head
[ "$(tree_of "$NEW")" = "$(tree_of "$C")" ] || die "tree of $NEW differs from tree of $C"
git diff --quiet "$C" "$NEW" || die "git diff $C $NEW is not empty"
for x in "$C" ${A:+"$A"} ${U:+"$U"}; do
  git merge-base --is-ancestor "$x" "$NEW" || die "$x is not an ancestor of $NEW"
done

echo "case:      $case_name"
echo "remote:    $remote $branch = ${A:-<none>}"
echo "record:    $gate (candidate $rec_candidate, status $rec_status)"
echo "candidate: $C (tree $(tree_of "$C"))"
echo "new head:  $NEW (tree $(tree_of "$NEW"))"
echo "push:      git push $remote $NEW:refs/heads/$branch"

if [ "$push" != 1 ]; then
  echo "DRY RUN: nothing pushed (pass --push)"
  exit 0
fi
now=$(git ls-remote "$remote" "refs/heads/$branch" | cut -f1)
[ "$now" = "$A" ] || die "remote $branch moved from ${A:-<none>} to ${now:-<none>} meanwhile; re-check coverage and rerun"
git push "$remote" "$NEW:refs/heads/$branch"
after=$(git ls-remote "$remote" "refs/heads/$branch" | cut -f1)
[ "$after" = "$NEW" ] || die "after the push the remote $branch is $after, not $NEW"
echo "PUSHED $branch = $NEW"
