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
# Exit codes: 0 clean and in sync, 1 uncommitted changes, 2 not pushed (the
#             remote has no such branch, or an older commit on it), 3 could not
#             reach the remote.

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
    -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

# Only a real revision counts. gh prints its API error body on *stdout* -- a 422
# for a branch the remote has never seen -- so anything that is not 40 hex
# characters is a message, not a sha, and must not be compared against HEAD.
is_sha() { [[ "${1:-}" =~ ^[0-9a-f]{40}$ ]]; }

# Sets REMOTE_SHA and REMOTE_STATE: present | absent | unreachable. "absent" is
# the case that used to be indistinguishable from a broken ssh-agent: git
# reached the remote perfectly well and reported that the branch is not there.
read_remote_sha() {
  local refs sha
  REMOTE_SHA=""
  REMOTE_STATE="unreachable"
  if refs="$(git ls-remote origin "refs/heads/$BRANCH" 2>/dev/null)"; then
    REMOTE_SHA="$(printf '%s\n' "$refs" | cut -f1)"
    if is_sha "$REMOTE_SHA"; then
      REMOTE_STATE="present"
    else
      REMOTE_SHA=""
      REMOTE_STATE="absent"
    fi
    return
  fi
  # git could not reach it (a locked ssh-agent does this). gh may still work.
  if command -v gh >/dev/null 2>&1 \
     && sha="$(gh api "repos/{owner}/{repo}/commits/$BRANCH" --jq .sha 2>/dev/null)" \
     && is_sha "$sha"; then
    REMOTE_SHA="$sha"
    REMOTE_STATE="present"
  fi
}

push_branch() {
  local slug
  say "Pushing $BRANCH..."
  if git push -q origin "$BRANCH" 2>/dev/null; then return 0; fi
  say "  SSH push failed; retrying over https with the gh credential helper"
  slug="$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)"
  # Same trap as the sha above: a failed gh prints its error body on stdout.
  [[ "$slug" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || return 1
  git -c credential.helper='!gh auth git-credential' push -q \
    "https://github.com/$slug.git" "$BRANCH" 2>/dev/null
}

read_remote_sha

# A branch the remote has never seen needs pushing too, which is what `-c` is
# documented to do: "commit anything pending, push, verify".
if [[ ( "$REMOTE_STATE" != "present" || "$local_sha" != "$REMOTE_SHA" ) && -n "$COMMIT_MSG" ]]; then
  if push_branch; then
    read_remote_sha
  fi
fi

# --- 3. verdict ------------------------------------------------------------- #
status=0
[[ $dirty -eq 1 ]] && status=1
if [[ $status -eq 0 ]]; then
  if [[ "$REMOTE_STATE" == "unreachable" ]]; then
    status=3
  elif [[ "$REMOTE_STATE" != "present" || "$local_sha" != "$REMOTE_SHA" ]]; then
    status=2
  fi
fi

if [[ $status -eq 0 ]]; then
  say "Clean and in sync: $BRANCH at ${local_sha:0:7}"
else
  [[ $dirty -eq 1 ]] && say "!  Working tree is dirty."
  if [[ "$REMOTE_STATE" == "unreachable" ]]; then
    say "!  Could not reach the remote to confirm the push."
    say "   (ssh-agent may be empty; 'gh auth setup-git' or push over https)"
  elif [[ "$REMOTE_STATE" == "absent" ]]; then
    say "!  $BRANCH is not on the remote yet."
    say "   (scripts/git-sync.sh -c \"message\" commits and pushes it)"
  elif [[ "$local_sha" != "$REMOTE_SHA" ]]; then
    say "!  Local ${local_sha:0:7} != remote ${REMOTE_SHA:0:7}"
  fi
fi
exit $status
