# italy-rs

Native volumetric pathtracing editor/renderer. See `humans.md` for
prerequisites, build/run instructions, and phase status; it also links the
full design plan for architecture and phased scope.

Standalone project — not part of `s-rank`, no wiring into its dashboard,
`agents.toon`, or GPU-mutex scripts. It happens to share this machine's one
GPU with `s-rank`'s SUPIR/TripoSplat services; running them at the same time
can OOM either side, but there's no code-level mutex between them (italy is
an interactive foreground app, not a daemon).

## Doctrine: beauty > performance > realism

This is a ranked ladder, not a list of nice-to-haves. When two of these
conflict, the higher number yields to the lower one. Cite it by number in
review.

1. **Beauty.** The image is the product. A cleaner, better-graded,
   better-framed image beats a more defensible one. "Technically correct but
   noisy/ugly" is a bug report, not a defense.
2. **Performance.** Beauty you can't iterate on isn't beauty. Interactive
   preview convergence is a feature, and it is the budget that constrains
   (1) — an effect nobody will wait for doesn't make anything beautiful.
3. **Realism.** Physical accuracy is a *means* to (1), not a goal in itself.
   Where it serves the image it stays; where it costs (1) or (2) it yields.
   This project is not a light-transport validator.

Corollaries, so the ladder doesn't get misread:

- **Unbiased transport is kept for (1), not for (3).** Energy-conserving
  BSDFs, correct MIS weights, and correct pdfs are the cheapest known route
  to a clean image — break them and you get noise and colour casts, which
  cost (1) directly. So "beauty over realism" is never license to fudge the
  estimator. It *is* license to add a firefly clamp on top of it.
- **Artist controls are features, not cheats.** Exposure, look strength,
  aperture, focus, environment rotation, light size/colour: these serve (1)
  and need no physical justification beyond looking better.
- **Mark every deliberate departure.** `src/` and `tests/` carry no
  comments — not even `realism:`/`claudia:` markers (2026-09-02 convention
  change; the code should read clearly enough not to need them, and the
  record lives in one place instead of scattered across files). Any place
  physics was traded away on purpose, or a minimalism-ladder rung was
  deliberately skipped, gets one line in `humans.md`'s Status section
  instead: `realism: <what was traded> — <why it looks better>` or
  `claudia: <ceiling> — upgrade if <trigger>`. Greppable there
  (`grep -n "^realism:\|^claudia:" humans.md`). Unrecorded non-physical
  behaviour is a bug, not a style choice.
- **When in doubt, render it both ways and look.** Verification in this
  repo is visual by design (`ITALY_DUMP_FRAME`, see `humans.md`). An
  argument from first principles loses to a side-by-side.

Read before touching `src/`:
- Linux + Windows, NVIDIA-only (CUDA + OptiX). No AMD/Vulkan support.
- **The GPU seam is `src/render/optix_renderer.h`** — one concrete class,
  not the multi-backend Device/Buffer/AccelStructure interface split the
  design doc sketches. `app/` and `core/` talk to it and to nothing else
  vendor-specific.
- There are exactly **two** sanctioned OptiX call sites:
  `src/render/` (the renderer) and `src/convert/sdf_baker.cpp` (a one-shot
  load-time bake that stands up its own short-lived context — see the
  sdf_baker.cpp marker in humans.md's Status section for why it isn't
  routed through the renderer). Adding a third needs a reason in the same
  form. `io/` and `app/` include neither `optix.h` nor `cuda_runtime.h`.
- A `src/rhi/` split with real per-backend interfaces is **deferred, not
  forgotten**: it happens the day a second backend (Metal) actually starts,
  and not before — there is nothing to abstract with one backend and one
  call site. Marked in humans.md's Status section (optix_renderer.h entry).
- Build: `cmake -B build -G Ninja && cmake --build build`.

## Agent context docs (`docs/agent/`)

Cold-start orientation without reading full prose:
- `filemap.toon` — every `src/`/`tests/` file: path, one-line purpose (from
  its header comment, where one exists — since the no-comments convention
  above, that's usually blank; `humans.md` is the purpose source of truth
  now), local `#include` edges. Regenerate anytime with
  `python3 scripts/gen_filemap.py` from the repo root — pure Python over
  `src/`/`tests/`, no CMake config or build needed, so it's cheap to rerun
  mid-iteration for a fresh file map or a quick seam check without paying
  for a full OptiX/CUDA build. The `agent_filemap` CMake target runs the
  same script as an `ALL` build step so a built tree is never stale either
  way — never hand-edit `filemap.toon` itself. Also flags any file outside
  `src/render/` and `src/convert/sdf_baker.cpp` that includes
  `optix.h`/`cuda_runtime.h` (the seam rule above): a nonzero exit
  standalone, a build-time error under CMake.
- `status.toon` / `commands.toon` — compact phase-status table and
  build/run/test/env-var command reference. Hand-maintained, not
  generated (source is `humans.md` prose, not code) — update these when
  `humans.md`'s Status/Build/Quick-start sections change.

Considered clangd/LSP instead of the generated file map: more accurate, but
needs a running server and an MCP/tool bridge this project has no access
to, and pays indexing cost on every cold start. Not worth it at ~40 source
files with one OptiX call-site to track — see the `claudia:` marker at the
top of `scripts/gen_filemap.py` for the upgrade trigger.

Format: [TOON](https://github.com/toon-format/spec) (`NAME[count]{fields}:`
tabular header, comma rows) — checked against benchmarks before adopting:
~40% fewer tokens than JSON on uniform tabular data like this, and its
explicit field schema + row count catch truncation/drift a plain CSV
wouldn't. `gen_filemap.py`'s writer follows the real spec's quoting rules
(§7.1/§7.2), not an ad hoc guess.
