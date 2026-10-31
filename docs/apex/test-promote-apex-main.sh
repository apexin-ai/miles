#!/bin/bash
# Toy-repository test of promote-apex-main.sh. Uses only a local bare repository as the remote.
#   bash docs/apex/test-promote-apex-main.sh <scratch dir, will be wiped> [promoter script]
# Exits non-zero on the first failed assertion. Without a promoter argument it also checks its own sensitivity:
# rerun with a promoter that always exits 42, it must fail.
set -u
[ $# -ge 1 ] || { echo "usage: $0 <scratch dir> [promoter]" >&2; exit 2; }
S=$(mkdir -p "$1" && cd "$1" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
P=${2:-$HERE/promote-apex-main.sh}
P=$(cd "$(dirname "$P")" && pwd)/$(basename "$P")
[ -f "$P" ] || { echo "promoter not found: $P" >&2; exit 2; }

fail() { echo "FAIL: $*" >&2; exit 1; }
checks=0
ok() { checks=$((checks + 1)); echo "ok   $*"; }

rm -rf "$S/t" && mkdir -p "$S/t" && cd "$S/t" || exit 2
git init -q --bare remote.git && git init -q -b up w && cd w || exit 2
g() { git -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }
git remote add origin "$S/t/remote.git"
remote_main() { git ls-remote origin refs/heads/apex/main | cut -f1; }
record() { # record <file> <candidate> <status> [extra line]
  { echo "# acceptance record"; echo "candidate: $2"; echo "status: $3"; [ $# -ge 4 ] && echo "$4"; } > "$1"
}
run() { bash "$P" "$@" > "$S/t/out.txt" 2>&1; }
expect_ok() { # expect_ok <label> <args...>
  local label=$1; shift
  run "$@" || { cat "$S/t/out.txt" >&2; fail "$label: expected exit 0"; }
  ok "$label"
}
expect_refused() { # expect_refused <label> <pattern> <args...>; the remote must not change
  local label=$1 pattern=$2 before; shift 2
  before=$(remote_main)
  if run "$@"; then cat "$S/t/out.txt" >&2; fail "$label: expected a refusal"; fi
  grep -q "ABORT: .*$pattern" "$S/t/out.txt" || { cat "$S/t/out.txt" >&2; fail "$label: refusal does not mention '$pattern'"; }
  [ "$(remote_main)" = "$before" ] || fail "$label: the remote changed on a refusal"
  ok "$label"
}
eq() { [ "$1" = "$2" ] || fail "$3: '$1' != '$2'"; ok "$3"; }

echo b > base.txt && g add . && g commit -qm U0 && U0=$(git rev-parse HEAD)
git checkout -q -b cand
echo c1 > c.txt && g add . && g commit -qm C1 && C1=$(git rev-parse HEAD)

# ---- acceptance record contract (CRV-FIX-031)
R=$S/t/rec
R1=$R-1 R2=$R-2 RL=$R-L R3=$R-3 R4=$R-4 R5=$R-5
expect_refused "missing record" "not found" --candidate "$C1" --gate "$R-none"
for st in FAILED NOT_RUN "NOT RUN" PENDING passed; do
  record "$R-st" "$C1" "$st"; expect_refused "status $st" "not PASSED" --candidate "$C1" --gate "$R-st" --push
done
record "$R-sha" "$U0" PASSED; expect_refused "record for another SHA" "not for the candidate" --candidate "$C1" --gate "$R-sha" --push
record "$R-short" "${C1:0:12}" PASSED; expect_refused "short SHA" "full 40-hex" --candidate "$C1" --gate "$R-short" --push
record "$R-mixed" "$C1" PASSED "status: FAILED"; expect_refused "two status lines" "more than once" --candidate "$C1" --gate "$R-mixed" --push
record "$R-twoc" "$C1" PASSED "candidate: $U0"; expect_refused "two candidate lines" "more than once" --candidate "$C1" --gate "$R-twoc" --push
record "$R-key" "$C1" PASSED "verdict: maybe"; expect_refused "unknown key" "unknown key" --candidate "$C1" --gate "$R-key" --push
echo "gate passed for $C1" > "$R-log"; expect_refused "free-text log naming the SHA" "malformed" --candidate "$C1" --gate "$R-log" --push
printf 'candidate: %s\n' "$C1" > "$R-nostatus"; expect_refused "no status line" "no status" --candidate "$C1" --gate "$R-nostatus" --push

# ---- first creation, idempotence, fast-forward
record "$R1" "$C1" PASSED "gate: toy"
expect_ok "first creation, dry run" --candidate "$C1" --gate "$R1" --upstream "$U0"
eq "$(remote_main)" "" "dry run pushes nothing"
expect_ok "first creation" --candidate "$C1" --gate "$R1" --upstream "$U0" --push
eq "$(remote_main)" "$C1" "apex/main = C1"
expect_ok "same candidate again" --candidate "$C1" --gate "$R1" --push
grep -q "nothing to do" "$S/t/out.txt" || fail "same candidate again: expected 'nothing to do'"
eq "$(remote_main)" "$C1" "retry keeps apex/main = C1"
echo c2 >> c.txt && g commit -qam C2 && C2=$(git rev-parse HEAD); record "$R2" "$C2" PASSED
expect_ok "fast-forward to C2" --candidate "$C2" --gate "$R2" --push
eq "$(remote_main)" "$C2" "apex/main = C2"
git checkout -q -b mainlocal "$C2"; echo legacy > legacy_impl.py && g add . && g commit -qm legacy && L=$(git rev-parse HEAD)
record "$RL" "$L" PASSED; expect_ok "fast-forward to L (local legacy fix)" --candidate "$L" --gate "$RL" --push
eq "$(remote_main)" "$L" "apex/main = L"
expect_refused "promoting an older candidate" "would go back" --candidate "$C2" --gate "$R2" --push

# ---- new upstream: adopt the candidate tree (CRV-FIX-032 retry included)
git checkout -q -b up1 "$U0"; echo b2 > base.txt; echo up > upstream_impl.py; g add . && g commit -qm U1 && U1=$(git rev-parse HEAD)
git checkout -q -b cand3 "$U1"; echo c1 > c.txt; echo c2 >> c.txt; g add . && g commit -qm C3 && C3=$(git rev-parse HEAD)
record "$R3" "$C3" PASSED
expect_refused "adopt without --coverage-checked" "coverage-checked" --candidate "$C3" --gate "$R3" --upstream "$U1"
expect_refused "upstream not an ancestor" "not an ancestor of the candidate" --candidate "$C3" --gate "$R3" --upstream "$C2" --coverage-checked
expect_ok "adopt, dry run" --candidate "$C3" --gate "$R3" --upstream "$U1" --coverage-checked
eq "$(remote_main)" "$L" "dry run keeps apex/main = L"
expect_ok "adopt" --candidate "$C3" --gate "$R3" --upstream "$U1" --coverage-checked --push
git fetch -q origin; M=$(remote_main)
eq "$(git rev-parse "$M^{tree}")" "$(git rev-parse "$C3^{tree}")" "tree(apex/main) = tree(C3)"
eq "$(git log -1 --format=%P "$M" | wc -w | tr -d ' ')" 2 "the adopt commit has two parents"
eq "$(git log -1 --format=%P "$M")" "$L $C3" "its parents are the old apex/main and C3"
eq "$(git ls-tree --name-only "$M" | grep -c legacy_impl)" 0 "the legacy file is gone"
git merge-base --is-ancestor "$U1" "$M" || fail "U1 is not an ancestor of apex/main"; ok "U1 is an ancestor"
expect_ok "adopt the same candidate again" --candidate "$C3" --gate "$R3" --upstream "$U1" --coverage-checked --push
grep -q "nothing to do" "$S/t/out.txt" || fail "adopt again: expected 'nothing to do'"
eq "$(remote_main)" "$M" "retry keeps the same apex/main"

# same tree, unrelated history: not adopted yet, must be adopted (not skipped on tree equality alone)
git checkout -q --orphan unrelated "$M"; g commit -qm "same tree, no shared history" && C4=$(git rev-parse HEAD)
eq "$(git rev-parse "$C4^{tree}")" "$(git rev-parse "$M^{tree}")" "C4 has the tree of apex/main"
record "$R4" "$C4" PASSED
expect_refused "same tree, unrelated: needs --coverage-checked" "coverage-checked" --candidate "$C4" --gate "$R4"
expect_ok "same tree, unrelated: adopt" --candidate "$C4" --gate "$R4" --coverage-checked --push
git fetch -q origin; M4=$(remote_main)
[ "$M4" != "$M" ] || fail "same tree, unrelated: apex/main did not move"
git merge-base --is-ancestor "$C4" "$M4" || fail "C4 is not an ancestor after the adopt"; ok "C4 is an ancestor after the adopt"

# a real update on top of apex/main
git checkout -q -b cand5 "$M4"; echo c5 > c5.txt && g add . && g commit -qm C5 && C5=$(git rev-parse HEAD)
record "$R5" "$C5" PASSED; expect_ok "real update fast-forwards" --candidate "$C5" --gate "$R5" --push
eq "$(remote_main)" "$C5" "apex/main = C5"

echo "PASS: $checks checks"

# ---- sensitivity: the same driver with a promoter that always fails must fail
if [ $# -lt 2 ]; then
  stub=$S/always-42.sh; printf '#!/bin/bash\nexit 42\n' > "$stub"
  if bash "$0" "$S/sensitivity" "$stub" > "$S/sensitivity.log" 2>&1; then
    fail "sensitivity: the driver passed with a promoter that always exits 42"
  fi
  echo "ok   sensitivity: the driver fails with a promoter that always exits 42"
fi
