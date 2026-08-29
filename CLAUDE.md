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
