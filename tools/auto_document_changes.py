#!/usr/bin/env python3
"""Automatically append important staged changes to docs/IMPORTANT_CHANGES.md."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_IMPORTANT_PREFIXES = (
    "core/",
    "analyzers/",
    "main.py",
    "config.py",
    "benchmarks/reference_suite.py",
)

DEFAULT_IMPORTANT_EXTRA_FILES = {
    "requirements.txt",
    "README.md",
    "AGENTS.md",
    ".github/copilot-instructions.md",
}

LOG_FILE = Path("docs/IMPORTANT_CHANGES.md")
IMPORTANT_RULES_FILE = Path("tools/auto_doc_important_paths.txt")


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def get_repo_root() -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"]))


def staged_files() -> list[str]:
    output = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    if not output:
        return []
    return [
        line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()
    ]


def working_tree_files() -> list[str]:
    output = run_git(["diff", "--name-only", "--diff-filter=ACMR"])
    if not output:
        return []
    return [
        line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()
    ]


def load_important_rules(repo_root: Path) -> list[str]:
    path = repo_root / IMPORTANT_RULES_FILE
    if path.exists():
        lines = [
            line.strip().replace("\\", "/")
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        rules = [line for line in lines if line and not line.startswith("#")]
        if rules:
            return rules

    rules = list(DEFAULT_IMPORTANT_PREFIXES)
    rules.extend(sorted(DEFAULT_IMPORTANT_EXTRA_FILES))
    return rules


def is_important(path: str, rules: list[str]) -> bool:
    normalized = path.replace("\\", "/")
    for rule in rules:
        if rule.endswith("/"):
            if normalized.startswith(rule):
                return True
        elif normalized == rule:
            return True
    return False


def build_fingerprint(files: list[str]) -> str:
    h = hashlib.sha1()
    for file_path in sorted(files):
        h.update(file_path.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def ensure_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Important Changes\n\n"
        "Automatically generated entries from local automation (git hooks + agents).\n"
        "Only important file changes are listed.\n\n"
    )
    path.write_text(header, encoding="utf-8")


def resolve_actor(actor_arg: str) -> str:
    if actor_arg and actor_arg != "auto":
        return actor_arg

    actor_env = os.environ.get("AUTO_DOC_ACTOR", "").strip()
    if actor_env:
        return actor_env

    try:
        actor_cfg = run_git(["config", "--get", "autodoc.actor"]).strip()
    except Exception:
        actor_cfg = ""
    if actor_cfg:
        return actor_cfg

    return "unknown"


def append_entry(
    path: Path,
    important_files: list[str],
    fingerprint: str,
    mode: str,
    actor: str,
) -> bool:
    ensure_header(path)
    content = path.read_text(encoding="utf-8")
    marker = f"<!-- autodoc:{fingerprint} -->"
    if marker in content:
        return False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"## {timestamp}",
        marker,
        "",
        f"Source: `{mode}` by `{actor}`",
        "",
        "Affected critical paths:",
    ]
    lines.extend([f"- `{file_path}`" for file_path in sorted(important_files)])
    lines.append("")

    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--staged", action="store_true", help="Document staged changes."
    )
    parser.add_argument(
        "--working-tree",
        action="store_true",
        help="Document modified files in working tree (agent-friendly, no commit required).",
    )
    parser.add_argument(
        "--actor",
        default="auto",
        help="Actor label for the entry (default: auto detection via env/config).",
    )
    parser.add_argument(
        "--filter-important-stdin",
        action="store_true",
        help="Read file paths from stdin and print only important ones (for hooks).",
    )
    args = parser.parse_args()

    mode = "staged"
    if args.working_tree:
        mode = "working-tree"
    elif args.staged:
        mode = "staged"

    repo_root = get_repo_root()
    rules = load_important_rules(repo_root)

    if args.filter_important_stdin:
        incoming = [
            line.strip().replace("\\", "/")
            for line in sys.stdin.read().splitlines()
            if line.strip()
        ]
        filtered = [path for path in incoming if is_important(path, rules)]
        if filtered:
            print("\n".join(sorted(set(filtered))))
        return 0

    changed_files = working_tree_files() if mode == "working-tree" else staged_files()
    if not changed_files:
        return 0

    important = [path for path in changed_files if is_important(path, rules)]
    if not important:
        return 0

    fingerprint = build_fingerprint(important)
    actor = resolve_actor(args.actor)
    log_path = repo_root / LOG_FILE
    changed = append_entry(log_path, important, fingerprint, mode=mode, actor=actor)
    if changed:
        print(
            f"[auto-doc] Updated {LOG_FILE.as_posix()} ({len(important)} important file(s), source={mode}, actor={actor})."
        )
    else:
        print(
            f"[auto-doc] Entry already present for current important changes (source={mode})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
