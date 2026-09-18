#!/usr/bin/env python3
"""Build an OCI layer tar representing what docker/triton/Dockerfile's RUN
steps changed, relative to the NGC base image.

Used by docker/README.md's "Building the triton image" fallback path, for
hosts that cannot run Docker at all -- which is the situation this project is
in: Runpod pods block the unshare()/clone() syscalls nested containers need,
and GitHub-hosted runners cannot fit the 22GB base image. The image currently
published as ghcr.io/raoashish10/speech-cascade-triton:latest was built this
way and verified by running it.

Prerequisites: run the Dockerfile's RUN and COPY steps natively on a pod using
the same base image, then produce the base image's flattened file listing with

    crane export <base-image> - | tar -tf - > /tmp/base-files.txt

This then writes /tmp/layer.tgz's file lists, to be packaged with

    tar -cf - -C / -T /tmp/layer-files.txt -C /tmp/wh-stage -T /tmp/layer-whiteouts.txt | pigz -1 > /tmp/layer.tgz

and appended with `crane mutate --append`.

This reproduces what Docker's overlay driver would produce for those RUN
layers. Two kinds of entry:

  - ADDED/MODIFIED files: everything under the touched roots whose mtime is
    newer than the base image's build date. pip writes install-time mtimes,
    and the base image's files are all stamped at image build time, so this
    separates cleanly (verified: base files are Mar 2026, ours are Sep 2026).

  - DELETED files: present in the base image's flattened listing but gone
    now, because `pip install` uninstalls the version it replaces. These need
    explicit `.wh.<name>` whiteout markers or the base layer's copy resurfaces
    -- which for a dist-info directory means importlib.metadata seeing two
    versions of the same package. A directory deleted in full collapses to a
    single whiteout rather than one per child.
"""
import os
import subprocess
import sys

ROOTS = [
    "opt/venv-tritonserver",
    "venv",
    "workspace/speech-cascade-inference",
    "usr/local/bin/entrypoint.sh",
]
CUTOFF = "2026-06-01"          # between the base image build and our install
STAGE = "/tmp/wh-stage"        # where .wh. markers are materialized
OUT_LIST = "/tmp/layer-files.txt"


def current_files():
    """Every path that exists now under ROOTS (files, dirs, symlinks)."""
    out = set()
    for root in ROOTS:
        p = "/" + root
        if not os.path.exists(p):
            continue
        if os.path.isfile(p) or os.path.islink(p):
            out.add(root)
            continue
        for dirpath, dirnames, filenames in os.walk(p, followlinks=False):
            rel = os.path.relpath(dirpath, "/")
            out.add(rel)
            for n in dirnames + filenames:
                out.add(os.path.join(rel, n))
    return out


def base_files():
    """Base image paths under ROOTS, from the streamed tar listing."""
    out = set()
    with open("/tmp/base-files.txt") as fh:
        for line in fh:
            p = line.strip().lstrip("./").rstrip("/")
            if not p:
                continue
            if any(p == r or p.startswith(r + "/") for r in ROOTS):
                out.add(p)
    return out


def changed_files():
    """Added or modified: mtime newer than the base image build."""
    paths = [("/" + r) for r in ROOTS if os.path.exists("/" + r)]
    res = subprocess.run(
        ["find", *paths, "-newermt", CUTOFF, "-not", "-type", "d", "-print0"],
        capture_output=True,
    )
    return [p.lstrip("/") for p in res.stdout.decode("utf-8", "replace").split("\0") if p]


def main():
    cur, base = current_files(), base_files()
    deleted = base - cur

    # Collapse a fully-deleted directory into one whiteout for the directory.
    deleted_dirs = {d for d in deleted if not any(
        x.startswith(d + "/") for x in cur)}
    collapsed = set()
    for d in sorted(deleted_dirs, key=len):
        if not any(d.startswith(p + "/") for p in collapsed):
            collapsed.add(d)
    whiteouts = sorted(collapsed)

    changed = changed_files()

    print(f"base paths under roots : {len(base)}")
    print(f"current paths          : {len(cur)}")
    print(f"added/modified files   : {len(changed)}")
    print(f"whiteouts (deletions)  : {len(whiteouts)}")
    for w in whiteouts[:15]:
        print("   wh:", w)
    if len(whiteouts) > 15:
        print(f"   ... and {len(whiteouts) - 15} more")

    # Materialize .wh. markers
    subprocess.run(["rm", "-rf", STAGE], check=True)
    wh_rel = []
    for w in whiteouts:
        d, name = os.path.split(w)
        os.makedirs(os.path.join(STAGE, d), exist_ok=True)
        marker = os.path.join(d, ".wh." + name)
        open(os.path.join(STAGE, marker), "w").close()
        wh_rel.append(marker)

    with open(OUT_LIST, "w") as fh:
        fh.write("\n".join(changed) + "\n")
    with open("/tmp/layer-whiteouts.txt", "w") as fh:
        fh.write(("\n".join(wh_rel) + "\n") if wh_rel else "")
    print("wrote", OUT_LIST, "and /tmp/layer-whiteouts.txt")


if __name__ == "__main__":
    sys.exit(main())
