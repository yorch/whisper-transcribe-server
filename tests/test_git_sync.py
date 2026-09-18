"""Tests for scripts/git-sync.sh, the commit-and-verify guard.

The script exists because a green commit is not proof the tree is clean, so its
own exit codes have to be trustworthy. Two defects motivated these tests: a
branch the remote had never seen was reported as "could not reach the remote"
(and `-c` then exited 3 without pushing it), and a failed `gh` prints its API
error body on stdout, which the script captured and compared as if it were a
revision.

Every case runs in a throwaway repo with a bare local remote: no network, no
GitHub account, and the developer's git config is bypassed so a global
gpgsign or template hook cannot change the outcome.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "git-sync.sh"
GIT = shutil.which("git")
# The guard is a bash script, so exercising it needs bash. Windows runners have
# neither /bin/bash nor a bash on PATH, and skipping is the honest outcome there
# rather than a failure that says nothing about the script.
BASH = shutil.which("bash") or ("/bin/bash" if Path("/bin/bash").exists() else None)

pytestmark = pytest.mark.skipif(
    GIT is None or BASH is None,
    reason="the guard is a bash script; git and bash are both required",
)


def git_bin() -> str:
    assert GIT is not None
    return GIT


def run(argv: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        argv, cwd=cwd, env=env, capture_output=True, text=True
    )


def clean_env(**overrides: str) -> dict[str, str]:
    """The environment every git and guard call runs in.

    Global and system config are bypassed, and anything gh-related is dropped so
    a developer's session cannot decide what these tests assert.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GH_", "GITHUB_"))}
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    env.update(overrides)
    return env


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = run([git_bin(), "-C", str(repo), *args], repo, env or clean_env())
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout.strip()


class Guard:
    """A scratch checkout with a bare local remote and a copy of the script."""

    def __init__(self, tmp_path: Path) -> None:
        self.work = tmp_path / "work"
        self.remote = tmp_path / "remote.git"
        self.remote.mkdir()
        run(
            [git_bin(), "init", "-q", "--bare", str(self.remote)], tmp_path, clean_env()
        )
        run([git_bin(), "init", "-q", str(self.work)], tmp_path, clean_env())
        git(self.work, "config", "user.email", "guard@example.com")
        git(self.work, "config", "user.name", "Guard")
        git(self.work, "config", "commit.gpgsign", "false")
        git(self.work, "config", "tag.gpgsign", "false")
        git(self.work, "remote", "add", "origin", str(self.remote))
        (self.work / "scripts").mkdir()
        shutil.copy(SCRIPT, self.work / "scripts" / "git-sync.sh")
        (self.work / "file.txt").write_text("one\n", encoding="utf-8")
        git(self.work, "add", "-A")
        git(self.work, "commit", "-qm", "first")

    # -- helpers ----------------------------------------------------------- #

    @property
    def branch(self) -> str:
        return git(self.work, "rev-parse", "--abbrev-ref", "HEAD")

    def head(self) -> str:
        return git(self.work, "rev-parse", "HEAD")

    def push(self) -> None:
        git(self.work, "push", "-q", "origin", f"HEAD:refs/heads/{self.branch}")

    def remote_refs(self) -> dict[str, str]:
        out = git(self.work, "ls-remote", "origin")
        return {
            line.split("\t")[1]: line.split("\t")[0]
            for line in out.splitlines()
            if line.strip()
        }

    def make_unreachable(self) -> None:
        git(self.work, "remote", "set-url", "origin", "/nonexistent/nowhere.git")

    def guard(self, *args: str, env: dict[str, str] | None = None):
        assert BASH is not None
        return run(
            [BASH, str(self.work / "scripts" / "git-sync.sh"), *args],
            self.work,
            env or clean_env(),
        )

    def bin_without_gh(
        self, tmp_path: Path, gh_output: str | None = None, gh_exit: int = 1
    ) -> str:
        """A PATH holding only what the script needs, optionally with a fake gh.

        The stub prints `gh_output` on stdout and exits `gh_exit`: a failed gh
        api call prints its error body on stdout and exits non-zero, which is
        the trap this suite exists for.
        """
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        for name in ("git", "sed", "cut", "dirname", "which", "bash", "sh"):
            found = shutil.which(name) or (
                f"/bin/{name}" if Path(f"/bin/{name}").exists() else None
            )
            if found:
                link = bindir / name
                if not link.exists():
                    link.symlink_to(found)
        if gh_output is not None:
            stub = bindir / "gh"
            stub.write_text(
                f"#!/bin/sh\nprintf '%s\\n' '{gh_output}'\nexit {gh_exit}\n",
                encoding="utf-8",
            )
            stub.chmod(0o755)
        return str(bindir)


@pytest.fixture
def guard(tmp_path: Path) -> Guard:
    return Guard(tmp_path)


# --------------------------------------------------------------------------- #
# The three states the exit codes describe
# --------------------------------------------------------------------------- #


def test_in_sync_exits_zero(guard: Guard):
    guard.push()
    result = guard.guard()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Clean and in sync" in result.stdout


def test_uncommitted_changes_exit_one_and_name_the_file(guard: Guard):
    guard.push()
    (guard.work / "file.txt").write_text("changed\n", encoding="utf-8")

    result = guard.guard()

    assert result.returncode == 1
    assert "Uncommitted changes" in result.stdout
    assert "file.txt" in result.stdout


def test_a_commit_that_was_never_pushed_exits_two_with_real_shas(guard: Guard):
    guard.push()
    (guard.work / "file.txt").write_text("two\n", encoding="utf-8")
    git(guard.work, "commit", "-qam", "second")

    result = guard.guard()

    assert result.returncode == 2
    assert "!  Local " in result.stdout and " != remote " in result.stdout
    assert "{" not in result.stdout, "an error body must never be printed as a sha"


# --------------------------------------------------------------------------- #
# A branch the remote has never seen: the defect that motivated the work
# --------------------------------------------------------------------------- #


def test_a_branch_the_remote_has_never_seen_is_not_reported_as_unreachable(
    guard: Guard,
):
    result = guard.guard()

    assert result.returncode == 2, (
        "a branch that is absent is not pushed, not unreachable"
    )
    assert "is not on the remote yet" in result.stdout
    assert "Could not reach the remote" not in result.stdout


def test_commit_mode_pushes_a_branch_the_remote_has_never_seen(
    guard: Guard, tmp_path: Path
):
    """The regression that mattered: `-c` used to exit 3 without pushing."""
    (guard.work / "file.txt").write_text("two\n", encoding="utf-8")
    env = clean_env(PATH=guard.bin_without_gh(tmp_path))

    result = guard.guard("-c", "push a brand new branch", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Pushing " in result.stdout
    assert guard.remote_refs().get(f"refs/heads/{guard.branch}") == guard.head()


def test_commit_mode_also_pushes_when_the_branch_is_only_behind(
    guard: Guard, tmp_path: Path
):
    guard.push()
    (guard.work / "file.txt").write_text("two\n", encoding="utf-8")
    git(guard.work, "commit", "-qam", "second")

    result = guard.guard(
        "-c", "second", env=clean_env(PATH=guard.bin_without_gh(tmp_path))
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert guard.remote_refs().get(f"refs/heads/{guard.branch}") == guard.head()


# --------------------------------------------------------------------------- #
# An unreachable remote is its own state
# --------------------------------------------------------------------------- #


def test_an_unreachable_remote_exits_three(guard: Guard):
    guard.make_unreachable()

    result = guard.guard()

    assert result.returncode == 3
    assert "Could not reach the remote" in result.stdout


def test_commit_mode_on_an_unreachable_remote_still_exits_three(guard: Guard):
    (guard.work / "file.txt").write_text("two\n", encoding="utf-8")
    guard.make_unreachable()

    result = guard.guard("-c", "cannot go anywhere")

    assert result.returncode == 3, result.stdout + result.stderr
    assert "Could not reach the remote" in result.stdout


# --------------------------------------------------------------------------- #
# gh output is untrusted
# --------------------------------------------------------------------------- #


def test_a_gh_error_body_is_never_mistaken_for_a_sha(guard: Guard, tmp_path: Path):
    """gh prints its 422 body on stdout; it must not be compared against HEAD."""
    guard.make_unreachable()
    body = '{"message":"No commit found for SHA: nope","status":"422"}'
    env = clean_env(PATH=guard.bin_without_gh(tmp_path, gh_output=body, gh_exit=1))

    result = guard.guard(env=env)

    assert result.returncode == 3, result.stdout + result.stderr
    assert "message" not in result.stdout, "the error body leaked into the report"
    assert "Could not reach the remote" in result.stdout


def test_a_gh_sha_is_trusted_when_git_cannot_reach_the_remote(
    guard: Guard, tmp_path: Path
):
    """The fallback that exists on purpose: broken ssh, working gh."""
    guard.make_unreachable()
    env = clean_env(
        PATH=guard.bin_without_gh(tmp_path, gh_output=guard.head(), gh_exit=0)
    )

    result = guard.guard(env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Clean and in sync" in result.stdout


def test_a_gh_sha_that_differs_still_counts_as_unpushed(guard: Guard, tmp_path: Path):
    guard.make_unreachable()
    env = clean_env(PATH=guard.bin_without_gh(tmp_path, gh_output="0" * 40, gh_exit=0))

    result = guard.guard(env=env)

    assert result.returncode == 2, result.stdout + result.stderr
    assert "0000000" in result.stdout, "the comparison should show the short shas"
