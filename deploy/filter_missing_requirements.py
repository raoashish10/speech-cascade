#!/usr/bin/env python3
"""Print the requirement lines for packages a venv does NOT already have.

Used by docker/triton/Dockerfile to add chatterbox-tts's dependencies to
/venv/vllm without disturbing anything vLLM chose. vLLM pins a large, tightly
coupled stack -- torch, transformers, tokenizers, numpy, triton -- and
chatterbox has to live with those versions rather than the ones in
deploy/requirements-chatterbox.txt, which came from a box where chatterbox was
alone. Verified on a GPU pod: chatterbox-tts imports and synthesizes fine
against vLLM's numpy 2.3.5 and transformers 5.17.0, where the freeze asks for
1.26.4 and 5.2.0.

This is deliberately dynamic rather than a static exclusion list. vLLM's
transitive set is large (~200 packages) and changes with every release;
maintaining that list by hand would be a standing source of drift, and getting
it wrong means silently downgrading something vLLM needs.

    python3 deploy/filter_missing_requirements.py \
        deploy/requirements-chatterbox.txt /venv/vllm/bin/python3 \
        --also-exclude deploy/image-exclude-chatterbox.txt
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Never install these into the vLLM venv regardless of what the freeze says:
# vLLM owns the GPU stack, and a --no-deps install of a conflicting torch or
# CUDA wheel would break it in ways that surface far from the cause.
ALWAYS_SKIP_PREFIXES = ("torch", "nvidia-", "cuda-", "triton")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(line: str) -> str:
    return re.split(r"[=<>!~\[;@\s]", line, maxsplit=1)[0].strip()


def installed_packages(python: str) -> set[str]:
    out = subprocess.run(
        [python, "-m", "pip", "list", "--format=freeze"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {normalize(requirement_name(l)) for l in out.splitlines() if l.strip()}


def read_exclusions(path: str) -> set[str]:
    with open(path) as fh:
        return {
            normalize(entry)
            for entry in (l.split("#", 1)[0].strip() for l in fh)
            if entry
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("requirements")
    ap.add_argument("python", help="interpreter of the venv to check against")
    ap.add_argument("--also-exclude", default=None,
                    help="an image-exclude-*.txt of packages to drop as well")
    args = ap.parse_args()

    have = installed_packages(args.python)
    excluded = read_exclusions(args.also_exclude) if args.also_exclude else set()

    kept, skipped_installed, skipped_excluded = [], 0, 0
    for raw in open(args.requirements):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = normalize(requirement_name(line))
        if name.startswith(ALWAYS_SKIP_PREFIXES) or name in have:
            skipped_installed += 1
            continue
        if name in excluded:
            skipped_excluded += 1
            continue
        kept.append(line)

    print("\n".join(kept))
    print(
        f"filter_missing_requirements: adding {len(kept)}, "
        f"already present {skipped_installed}, excluded {skipped_excluded}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
