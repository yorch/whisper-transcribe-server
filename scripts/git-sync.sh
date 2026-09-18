#!/usr/bin/env bash
# Commit-and-verify guard for this repo.
#
# Exists because an automated formatter rewrites files *after* a commit, which
# once left a real change uncommitted while the commit itself looked complete.
# A green commit is not proof the tree is clean, so this checks both.
#
#   scripts/git-sync.sh                  report state; exit 1 if not clean+synced
#   scripts/git-sync.sh -c "message"     commit anything pending, push, verify
#   scripts/git-sync.sh -q               quiet: exit status only
#
# Exit codes: 0 clean and in sync, 1 uncommitted changes, 2 not pushed,
#             3 could not reach the remote.

set -euo pipefail

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
QUIET=0
COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -q|--quiet) QUIET=1; shift ;;
    -c|--commit) COMMIT_MSG="${2:-}"; [[ -n "$COMMIT_MSG" ]] || { echo "-c needs a message" >&2; exit 64; }; shift 2 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

say() { [[ $QUIET -eq 1 ]] || echo "$@"; }

# --- 1. uncommitted work ---------------------------------------------------- #
dirty=0
if [[ -n "$(git status --porcelain)" ]]; then
  dirty=1
  say "Uncommitted changes in the working tree:"
  [[ $QUIET -eq 1 ]] || git status --short | sed 's/^/  /'
fi

if [[ $dirty -eq 1 && -n "$COMMIT_MSG" ]]; then
  say "Committing..."
  git add -A
  git commit -q -m "$COMMIT_MSG"
  dirty=0
  say "  committed $(git rev-parse --short HEAD)"
fi

# --- 2. is it pushed? ------------------------------------------------------- #
local_sha="$(git rev-parse HEAD)"
remote_sha=""
# Prefer the configured remote; fall back to the gh credential helper, because
# a locked ssh-agent silently breaks the SSH path (seen in practice).
remote_sha="$(git ls-remote origin "refs/heads/$BRANCH" 2>/dev/null | cut -f1 || true)"
if [[ -z "$remote_sha" ]] && command -v gh >/dev/null 2>&1; then
  remote_sha="$(gh api "repos/{owner}/{repo}/commits/$BRANCH" --jq .sha 2>/dev/null || true)"
fi

if [[ -z "$remote_sha" ]]; then
  say "!  Could not reach the remote to confirm the push."
  say "   (ssh-agent may be empty; 'gh auth setup-git' or push over https)"
  [[ $dirty -eq 1 ]] && exit 1
  exit 3
fi

if [[ "$local_sha" != "$remote_sha" ]]; then
  if [[ -n "$COMMIT_MSG" ]]; then
    say "Pushing $BRANCH..."
    if ! git push origin "$BRANCH" >/dev/null 2>&1; then
      say "  SSH push failed; retrying over https with the gh credential helper"
      git -c credential.helper='!gh auth git-credential' \
        push "https://github.com/$(gh repo view --json nameWithOwner --jq .nameWithOwner).git" "$BRANCH" >/dev/null
    fi
    remote_sha="$(git ls-remote origin "refs/heads/$BRANCH" 2>/dev/null | cut -f1 || true)"
    [[ -z "$remote_sha" ]] && remote_sha="$(gh api "repos/{owner}/{repo}/commits/$BRANCH" --jq .sha 2>/dev/null || true)"
  fi
fi

# --- 3. verdict ------------------------------------------------------------- #
status=0
[[ $dirty -eq 1 ]] && status=1
[[ "$local_sha" != "$remote_sha" ]] && [[ $status -eq 0 ]] && status=2

if [[ $status -eq 0 ]]; then
  say "Clean and in sync: $BRANCH at ${local_sha:0:7}"
else
  [[ $dirty -eq 1 ]] && say "!  Working tree is dirty."
  [[ "$local_sha" != "$remote_sha" ]] && say "!  Local ${local_sha:0:7} != remote ${remote_sha:0:7}"
fi
exit $status
