#!/usr/bin/env python3
"""Generate docs/agent/filemap.toon from src/ and tests/.

Regex-parsed, not clang/libclang-backed — no compiler dependency, so this
runs standalone (no build dir needed) as well as wired into CMake.

// claudia: static regex-parsed map, not clangd/LSP-backed — upgrade if
// src/ grows past ~100 files or agents need real go-to-definition (this
// repo has no LSP-bridge tool available to drive that today anyway).

Also flags, as a side effect of already parsing includes, any file outside
render/ and convert/sdf_baker.cpp that includes optix.h or cuda_runtime.h —
CLAUDE.md's OptiX-seam rule.

// claudia: seam check is first-order only (a file's own #include lines) —
// doesn't follow includes transitively through local headers. Upgrade if
// a violation ever needs to hide behind an intermediate header to slip
// through.

TOON output follows the tabular-array form in the official spec
(https://github.com/toon-format/spec, `NAME[count]{fields}:` header, 2-space
indented comma rows, quoting per SPEC.md \xa77.2/\xa77.1) rather than an
ad hoc CSV-like guess.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ["src", "tests"]
SOURCE_EXTS = {".h", ".hpp", ".cpp", ".cu", ".cuh"}
SEAM_ALLOWED = {"src/render", "src/convert/sdf_baker.cpp"}
SEAM_HEADERS = {"optix.h", "cuda_runtime.h"}
PURPOSE_MAX_LEN = 200

PRAGMA_ONCE_RE = re.compile(r"^\s*#pragma\s+once\s*$")
BLOCK_COMMENT_RE = re.compile(r"^\s*/\*(.*?)\*/", re.DOTALL)
LINE_COMMENT_RE = re.compile(r"^\s*//(.*)$")
INCLUDE_LOCAL_RE = re.compile(r'^\s*#include\s*"([^"]+)"')
INCLUDE_ANGLE_RE = re.compile(r"^\s*#include\s*<([^>]+)>")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s")
NUMERIC_RE = re.compile(r"^[+-]?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?$", re.IGNORECASE)
CONTROL_RE = re.compile(r"[\x00-\x1f]")


def skip_leading_boilerplate(text: str) -> str:
    """Skip blank lines and a leading `#pragma once` so the doc comment
    underneath (the common case in this codebase) is actually reached."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (lines[i].strip() == "" or PRAGMA_ONCE_RE.match(lines[i])):
        i += 1
    return "\n".join(lines[i:])


def truncate_purpose(text: str, max_len: int = PURPOSE_MAX_LEN) -> str:
    text = text.strip()
    if not text:
        return text
    m = SENTENCE_END_RE.search(text)
    if m and m.start() <= max_len:
        return text[: m.start()].rstrip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "\u2026"


def extract_purpose(text: str) -> str:
    text = skip_leading_boilerplate(text)
    m = BLOCK_COMMENT_RE.match(text)
    if m:
        body_lines = m.group(1).splitlines()
    else:
        body_lines = []
        for line in text.splitlines():
            lm = LINE_COMMENT_RE.match(line)
            if not lm:
                break
            body_lines.append(lm.group(1))

    # First paragraph only: skip leading blank decorated lines, stop at the
    # first blank line once content has started.
    para = []
    started = False
    for raw in body_lines:
        line = raw.strip(" *\t")
        if not line:
            if started:
                break
            continue
        started = True
        para.append(line)
    return truncate_purpose(" ".join(para))


def extract_includes(text: str):
    local, angle = [], []
    for line in text.splitlines():
        m = INCLUDE_LOCAL_RE.match(line)
        if m:
            local.append(m.group(1))
            continue
        m = INCLUDE_ANGLE_RE.match(line)
        if m:
            angle.append(m.group(1))
    return local, angle


def needs_quoting(field: str, delimiter: str = ",") -> bool:
    """Per TOON SPEC.md \xa77.2."""
    if field == "":
        return True
    if field != field.strip(" \t"):
        return True
    if field in ("true", "false", "null"):
        return True
    if NUMERIC_RE.match(field):
        return True
    if any(c in field for c in ':"\\[]{}'):
        return True
    if CONTROL_RE.search(field):
        return True
    if delimiter in field:
        return True
    if field == "-" or field.startswith("-"):
        return True
    if field == "#" or field.startswith("#"):
        return True
    return False


def toon_escape(field: str, delimiter: str = ",") -> str:
    if not needs_quoting(field, delimiter):
        return field
    out = field.replace("\\", "\\\\").replace('"', '\\"')
    out = out.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    out = CONTROL_RE.sub(lambda m: f"\\u{ord(m.group(0)):04x}", out)
    return f'"{out}"'


def check_staleness():
    """Hand-maintained docs (status.toon/commands.toon) can't be
    regenerated — best we can do is warn when their source has moved on."""
    humans_md = REPO_ROOT / "humans.md"
    if not humans_md.exists():
        return
    humans_mtime = humans_md.stat().st_mtime
    for name in ("status.toon", "commands.toon"):
        p = REPO_ROOT / "docs" / "agent" / name
        if p.exists() and p.stat().st_mtime < humans_mtime:
            print(
                f"warning: docs/agent/{name} is older than humans.md and is "
                "hand-maintained — check it's still accurate",
                file=sys.stderr,
            )


def main() -> int:
    files = []
    for d in SCAN_DIRS:
        base = REPO_ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in SOURCE_EXTS:
                files.append(p)

    seam_violations = []
    rows = []
    for p in files:
        rel = p.relative_to(REPO_ROOT).as_posix()
        text = p.read_text(errors="replace")
        purpose = extract_purpose(text)
        local_inc, angle_inc = extract_includes(text)

        for h in list(angle_inc) + list(local_inc):
            if h in SEAM_HEADERS:
                allowed = any(
                    rel == a or rel.startswith(a + "/") for a in SEAM_ALLOWED
                )
                if not allowed:
                    seam_violations.append((rel, h))

        rows.append((rel, purpose, "|".join(local_inc)))

    out_dir = REPO_ROOT / "docs" / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "filemap.toon"

    lines = [f"files[{len(rows)}]{{path,purpose,includes}}:"]
    for rel, purpose, includes in rows:
        lines.append(
            f"  {toon_escape(rel)},{toon_escape(purpose)},{toon_escape(includes)}"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path.relative_to(REPO_ROOT)} ({len(rows)} files)")

    check_staleness()

    if seam_violations:
        print("OptiX-seam violations (CLAUDE.md: only src/render/ and "
              "src/convert/sdf_baker.cpp may include optix.h/cuda_runtime.h):",
              file=sys.stderr)
        for rel, h in seam_violations:
            print(f"  {rel} includes {h}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
