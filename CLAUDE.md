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
- **Mark every deliberate departure.** Any place physics was traded away on
  purpose gets a comment:

      // realism: <what was traded> — <why it looks better>

  Greppable the same way the `claudia:` markers are (`grep -rn "realism:"
  src/`). Unmarked non-physical behaviour is a bug, not a style choice.
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
  comment at the top of that file for why it isn't routed through the
  renderer). Adding a third needs a reason in the same form. `io/` and
  `app/` include neither `optix.h` nor `cuda_runtime.h`.
- A `src/rhi/` split with real per-backend interfaces is **deferred, not
  forgotten**: it happens the day a second backend (Metal) actually starts,
  and not before — there is nothing to abstract with one backend and one
  call site. Marked in `optix_renderer.h`.
- Build: `cmake -B build -G Ninja && cmake --build build`.

## Agent context docs (`docs/agent/`)

Cold-start orientation without reading full prose:
- `filemap.toon` — every `src/`/`tests/` file: path, one-line purpose (from
  its header comment, where one exists), local `#include` edges.
  Auto-regenerated on every build by the `agent_filemap` CMake target
  (`scripts/gen_filemap.py`) — never hand-edit, never goes stale relative
  to a built tree. Also flags any file outside `src/render/` and
  `src/convert/sdf_baker.cpp` that includes `optix.h`/`cuda_runtime.h` (the
  seam rule above) as a build-time error.
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
