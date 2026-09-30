#!/usr/bin/env python3
"""Pre-flight an image's layers for the packaging faults that make a pull fail.

Both faults below produce the same user-visible symptom: the pod never starts,
`runtime` stays null, and the pull RETRIES rather than failing fast, emitting
"Downloading" lines that look like slow progress. Both cost a pod deploy to
discover the hard way. This checks for them in seconds, against the registry,
before anything tries to run the image.

  xattrs      A layer packaged with BSD tar on macOS carries extended
              attributes the Linux runtime cannot set:
                failed to register layer: lsetxattr /workspace:
                xattr "com.apple.provenance": operation not supported
              Build layers on Linux, or use
              COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --no-acls.

  dangling    A hardlink entry whose target is not itself in the same
  hardlinks   archive:
                failed to register layer: link ...: no such file or directory
              Caused by listing a directory AND its contents (tar records the
              duplicate as a link), or by filtering one member of a hardlink
              set out of the file list. Use tar --hard-dereference.

Usage:
    python3 deploy/check_image_layers.py ghcr.io/owner/image:tag [n_layers]

n_layers defaults to 4 and limits the check to the TOP layers -- the ones
this repo adds. The NGC base image's layers are NVIDIA's and already known
good, and streaming all 85 of them would mean pulling ~16GB.
"""

import json
import subprocess
import sys
import tarfile


def crane(*args: str) -> bytes:
    return subprocess.run(["crane", *args], capture_output=True, check=True).stdout


def check_layer(image: str, digest: str, index: int) -> list[str]:
    """Stream one layer blob and report packaging faults."""
    blob = crane("blob", f"{image.split(':')[0].split('@')[0]}@{digest}")
    problems: list[str] = []
    names: set[str] = set()
    links: list[tuple[str, str]] = []
    xattr_members: list[str] = []
    appledouble: list[str] = []

    import io

    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
        for m in tf:
            names.add(m.name.lstrip("./"))
            if m.islnk():
                links.append((m.name, m.linkname.lstrip("./")))
            if any("xattr" in k.lower() for k in (m.pax_headers or {})):
                xattr_members.append(m.name)
            base = m.name.rsplit("/", 1)[-1]
            if base.startswith("._"):
                appledouble.append(m.name)

    if xattr_members:
        problems.append(
            f"{len(xattr_members)} member(s) carry xattr pax headers "
            f"(e.g. {xattr_members[:2]}) -- built on macOS?"
        )
    dangling = [(n, t) for n, t in links if t not in names]
    if dangling:
        problems.append(
            f"{len(dangling)} dangling hardlink(s) (e.g. {dangling[:2]}) -- "
            "target not in this layer"
        )
    if appledouble:
        problems.append(f"{len(appledouble)} AppleDouble file(s) (e.g. {appledouble[:2]})")

    status = "FAIL" if problems else "ok"
    print(f"  layer {index}: {len(names):>7} entries  {status}")
    for p in problems:
        print(f"      ! {p}")
    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    image = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    manifest = json.loads(crane("manifest", image))
    layers = manifest["layers"]
    top = layers[-n:]
    print(f"{image}\n{len(layers)} layers total; checking the top {len(top)}")

    failed = 0
    for i, layer in enumerate(top, start=len(layers) - len(top)):
        failed += len(check_layer(image, layer["digest"], i))

    print("\nFAILED -- this image will not pull" if failed else "\nall checked layers OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
