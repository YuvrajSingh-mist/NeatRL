#!/usr/bin/env bash
# Revert the Dueling-DQN changes this experiment made to NeatRL.
#
# What gets reverted:
#   1. src/neatrl/dueling_dqn_mlp.py   -- the `subtract_mean` switch and the
#      `adv_mean_subtraction` trainer argument. Restored to the paper's
#      hard-coded Q = V + A - mean(A).
#   2. tests/test_network_forward_pass.py -- the two extra forward-pass tests.
#
# What is kept (new files, not edits to your code):
#   scripts/dueling_qva_experiment.py
#   scripts/dueling_qva_changes/
#
# Usage:
#   ./scripts/dueling_qva_changes/revert.sh --dry-run   # check only
#   ./scripts/dueling_qva_changes/revert.sh             # actually revert
#
# How it works: it reverses `neatrl-dueling-changes.patch` with `git apply -R`.
# That only touches the hunks in the patch, and it *refuses* if the files no
# longer match the patched state -- so your own later edits are never silently
# dropped. Use `git checkout -- <files>` instead if you want a hard reset.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PATCH="scripts/dueling_qva_changes/neatrl-dueling-changes.patch"
DRY_RUN=0
case "${1:-}" in
  ""|--dry-run|--check) ;;
  *)
    echo "error: unknown argument '$1' (use --dry-run)" >&2
    exit 2
    ;;
esac
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "--check" ]]; then
  DRY_RUN=1
fi

if [[ ! -f "$PATCH" ]]; then
  echo "error: patch not found at $PATCH" >&2
  exit 1
fi

echo "Repo:   $REPO_ROOT"
echo "Patch:  $PATCH"
echo

echo "Files this patch touches:"
git --no-pager diff --stat HEAD -- src/neatrl/dueling_dqn_mlp.py \
  tests/test_network_forward_pass.py || true
echo

if ! git apply -R --check "$PATCH" 2>/tmp/revert_err; then
  echo "Cannot revert cleanly."
  echo
  cat /tmp/revert_err
  echo
  echo "The files no longer match the state this patch produced. Most likely"
  echo "you edited them further, or you already reverted. Options:"
  echo "  git checkout -- src/neatrl/dueling_dqn_mlp.py tests/test_network_forward_pass.py"
  echo "    (hard reset those two files to HEAD -- drops any of your edits too)"
  echo "  git diff -- src/neatrl/dueling_dqn_mlp.py    (inspect before deciding)"
  exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Patch reverses cleanly. Run without --dry-run to apply."
  exit 0
fi

git apply -R "$PATCH"
echo "Reverted:"
echo "  src/neatrl/dueling_dqn_mlp.py"
echo "  tests/test_network_forward_pass.py"
echo

STATUS="$(git status --short -- src/neatrl/dueling_dqn_mlp.py tests/test_network_forward_pass.py)"
if [[ -z "$STATUS" ]]; then
  echo "Both files are back to their committed state."
else
  echo "Remaining diff (should be unrelated edits only):"
  echo "$STATUS"
fi
echo
echo "Verify:"
echo "  grep -c subtract_mean src/neatrl/dueling_dqn_mlp.py   # expect 0"
