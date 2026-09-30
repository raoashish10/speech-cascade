#!/usr/bin/env python3
"""Drop packages named in an exclusion list from a pip-freeze requirements file.

Used by docker/triton/Dockerfile to install deploy/requirements-main.txt and
deploy/requirements-chatterbox.txt *minus* the packages the NGC base image
already ships. See deploy/image-exclude-main.txt for why that matters (short
version: those two files are `pip freeze` dumps from a CUDA 12.8 bare-metal
box, and blindly replaying them onto a CUDA 13.1 base image installed ~19GB
of duplicate CUDA userspace).

Name matching is PEP 503 normalized, so `tensorrt_llm`, `tensorrt-llm` and
`TensorRT-LLM` are all the same package and an exclusion list entry may be
spelled any of those ways.

By default an exclusion entry that matches nothing in the requirements file is
a hard error. That is deliberate: the exclusion lists encode "the base image
already provides this", and if a requirements file is later regenerated and a
package is renamed or dropped, we want the build to fail loudly rather than
silently start shipping a duplicate multi-GB CUDA stack again. Pass
--allow-unused to downgrade it to a warning.

    python3 deploy/filter_requirements.py \
        deploy/requirements-main.txt deploy/image-exclude-main.txt > filtered.txt
"""

from __future__ import annotations

import argparse
import re
import sys


def normalize(name: str) -> str:
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(line: str) -> str:
    """Extract the distribution name from one requirements-file line.

    Only has to cope with what `pip freeze` emits (name==version, plus the
    occasional extras marker), not the full PEP 508 grammar.
    """
    return re.split(r"[=<>!~\[;@\s]", line, maxsplit=1)[0].strip()


def read_exclusions(path: str) -> dict[str, str]:
    """Map normalized name -> the line it was spelled on, for error messages."""
    exclusions: dict[str, str] = {}
    with open(path) as handle:
        for raw in handle:
            entry = raw.split("#", 1)[0].strip()
            if entry:
                exclusions[normalize(entry)] = entry
    return exclusions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", help="pip-freeze style requirements file")
    parser.add_argument("exclusions", help="one package name per line, # comments ok")
    parser.add_argument(
        "--allow-unused",
        action="store_true",
        help="warn instead of failing when an exclusion matches no requirement",
    )
    args = parser.parse_args()

    exclusions = read_exclusions(args.exclusions)
    matched: set[str] = set()
    kept: list[str] = []

    with open(args.requirements) as handle:
        for raw in handle:
            stripped = raw.split("#", 1)[0].strip()
            if not stripped:
                continue
            key = normalize(requirement_name(stripped))
            if key in exclusions:
                matched.add(key)
                continue
            kept.append(stripped)

    unused = sorted(exclusions[k] for k in set(exclusions) - matched)
    if unused:
        message = (
            f"{args.exclusions}: {len(unused)} entr"
            f"{'y' if len(unused) == 1 else 'ies'} matched nothing in "
            f"{args.requirements}: {', '.join(unused)}"
        )
        if not args.allow_unused:
            print(f"ERROR: {message}", file=sys.stderr)
            print(
                "Either the requirements file was regenerated and this package "
                "is gone (drop the exclusion), or it was renamed (update the "
                "exclusion). Re-run with --allow-unused to proceed anyway.",
                file=sys.stderr,
            )
            return 1
        print(f"WARNING: {message}", file=sys.stderr)

    print("\n".join(kept))
    print(
        f"filter_requirements: kept {len(kept)}, dropped {len(matched)} "
        f"from {args.requirements}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
