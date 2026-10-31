#!/bin/bash
# Move apex/main to a validated candidate so that the tree of apex/main equals the tree of the candidate
# (BRANCH-PR plan section 5.3.2). Dry run by default; never force-pushes.
#
#   docs/apex/promote-apex-main.sh --candidate <C> --gate <result file naming C> [--upstream <U>]
#                                  [--coverage-checked] [--remote origin] [--branch apex/main] [--push]
#
# Cases, decided from the remote branch A (git ls-remote):
#   1. A does not exist          -> apex/main = C (first creation).
#   2. A is an ancestor of C     -> fast-forward apex/main to C.
#   3. otherwise (new upstream)  -> M = commit-tree(C^{tree}, parents A and C); needs --coverage-checked, i.e. every
#                                   local change of A is in C or was dropped with a written reason (docs/apex/patches.md).
# Asserted before any push: tree(new) == tree(C); A (if any), C and U (if given) are ancestors of the new head; the
# gate file names the full SHA of C; the remote branch is still A right before the push.
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
    -h | --help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { echo "ABORT: $*" >&2; exit 1; }
[ -n "$candidate" ] || die "--candidate is required"
[ -n "$gate" ] || die "--gate <file> is required (the gate result that validated the candidate)"
[ -f "$gate" ] || die "gate file not found: $gate"

git fetch -q "$remote"
C=$(git rev-parse --verify "$candidate^{commit}") || die "candidate is not a commit: $candidate"
grep -q "$C" "$gate" || die "gate file $gate does not name the full candidate SHA $C"
if [ -n "$upstream" ]; then
  U=$(git rev-parse --verify "$upstream^{commit}") || die "upstream is not a commit: $upstream"
  git merge-base --is-ancestor "$U" "$C" || die "upstream $U is not an ancestor of the candidate $C"
fi

A=$(git ls-remote "$remote" "refs/heads/$branch" | cut -f1)
if [ -z "$A" ]; then
  case_name="first creation"
  NEW=$C
elif [ "$A" = "$C" ]; then
  echo "$branch is already $C; nothing to do"
  exit 0
else
  git cat-file -e "$A^{commit}" 2>/dev/null || die "remote $branch = $A is not in the local clone (fetch it)"
  if git merge-base --is-ancestor "$A" "$C"; then
    case_name="fast-forward"
    NEW=$C
  else
    case_name="adopt candidate tree on top of $A"
    [ "$coverage" = 1 ] || die "$branch ($A) is not an ancestor of $C: check that every local change of $A is in $C or was dropped with a reason (docs/apex/patches.md), then pass --coverage-checked"
    NEW=$(git commit-tree "$C^{tree}" -p "$A" -p "$C" -m "apex/main: adopt validated candidate ${C:0:12}

The tree is exactly the tree of ${C:0:12}, which passed the gate in $(basename "$gate").
Parents: the previous apex/main ${A:0:12} and the candidate.${U:+ Upstream base: ${U:0:12}.}")
  fi
fi

# ---- assertions on the new head
[ "$(git rev-parse "$NEW^{tree}")" = "$(git rev-parse "$C^{tree}")" ] || die "tree of $NEW differs from tree of $C"
git diff --quiet "$C" "$NEW" || die "git diff $C $NEW is not empty"
for x in "$C" ${A:+"$A"} ${U:+"$U"}; do
  git merge-base --is-ancestor "$x" "$NEW" || die "$x is not an ancestor of $NEW"
done

echo "case:      $case_name"
echo "remote:    $remote $branch = ${A:-<none>}"
echo "candidate: $C (tree $(git rev-parse "$C^{tree}"))"
echo "new head:  $NEW (tree $(git rev-parse "$NEW^{tree}"))"
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
