#!/bin/bash
# toy test of promote-apex-main.sh: bash docs/apex/test-promote-apex-main.sh <empty scratch dir> docs/apex/promote-apex-main.sh
set -u
S=$1; P=$2
rm -rf "$S" && mkdir -p "$S" && cd "$S" && git init -q --bare remote.git && git init -q -b up w && cd w
g(){ git -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }
git remote add origin "$S/remote.git"
r(){ echo "--- $*" | sed -E "s#$S/##g; s/[0-9a-f]{40}/<sha>/g"; bash "$P" "$@" > "$S/out.txt" 2>&1; rc=$?; sed -E "s#$S/##g; s/[0-9a-f]{40}/<sha>/g" "$S/out.txt" | grep -E "ABORT|case:|PUSHED|DRY|nothing to do" | sed 's/^/    /'; echo "    rc=$rc"; }
echo b > base.txt && g add . && g commit -qm U0 && U0=$(git rev-parse HEAD)
git checkout -q -b cand
echo c1 > c.txt && g add . && g commit -qm C1 && C1=$(git rev-parse HEAD)
r --candidate "$C1" --gate /nonexistent
echo "gate for something else" > "$S/gate0.md"; r --candidate "$C1" --gate "$S/gate0.md"
echo "passed $C1" > "$S/gate1.md"
r --candidate "$C1" --gate "$S/gate1.md" --upstream "$U0"
r --candidate "$C1" --gate "$S/gate1.md" --upstream "$U0" --push
r --candidate "$C1" --gate "$S/gate1.md"
echo c2 >> c.txt && g commit -qam C2 && C2=$(git rev-parse HEAD); echo "passed $C2" > "$S/gate2.md"
r --candidate "$C2" --gate "$S/gate2.md" --push
git checkout -q -b mainlocal "$C2"; echo legacy > legacy_impl.py && g add . && g commit -qm legacy && L=$(git rev-parse HEAD); echo "passed $L" > "$S/gateL.md"
r --candidate "$L" --gate "$S/gateL.md" --push
git checkout -q -b up1 "$U0"; echo b2 > base.txt; echo up > upstream_impl.py; g add . && g commit -qm U1 && U1=$(git rev-parse HEAD)
git checkout -q -b cand3 "$U1"; echo c1 > c.txt; echo c2 >> c.txt; g add . && g commit -qm C3 && C3=$(git rev-parse HEAD); echo "passed $C3" > "$S/gate3.md"
r --candidate "$C3" --gate "$S/gate3.md" --upstream "$U1"
r --candidate "$C3" --gate "$S/gate3.md" --upstream "$C2" --coverage-checked
r --candidate "$C3" --gate "$S/gate3.md" --upstream "$U1" --coverage-checked
r --candidate "$C3" --gate "$S/gate3.md" --upstream "$U1" --coverage-checked --push
git fetch -q origin; M=$(git rev-parse origin/apex/main)
echo "remote tree==C3: $([ "$(git rev-parse "$M^{tree}")" = "$(git rev-parse "$C3^{tree}")" ] && echo yes || echo NO); legacy_impl.py in remote: $(git ls-tree --name-only "$M" | grep -c legacy); parents: $(git log -1 --format=%p "$M" | wc -w); old main L ancestor: $(git merge-base --is-ancestor "$L" "$M" && echo yes || echo no); C3 ancestor: $(git merge-base --is-ancestor "$C3" "$M" && echo yes || echo no)"
echo "plain merge would keep legacy: $(git checkout -q -b plain "$L" && g merge -q --no-edit "$C3" 2>/dev/null; ls legacy_impl.py 2>/dev/null | wc -l)"
