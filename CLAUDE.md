@/home/joe/projects/monkey-boy/CLAUDE.md.snippet

# italy

Native volumetric pathtracing editor/renderer. See `README.md` for setup and
the full design plan referenced there for architecture and phased scope.

Standalone project — not part of `s-rank`, no wiring into its dashboard,
`agents.toon`, or GPU-mutex scripts. It happens to share this machine's one
GPU with `s-rank`'s SUPIR/TripoSplat services; running them at the same time
can OOM either side, but there's no code-level mutex between them (italy is
an interactive foreground app, not a daemon).

Read before touching `src/`:
- Linux + Windows, NVIDIA-only (CUDA + OptiX). No AMD/Vulkan support.
- Vendor-specific GPU calls stay behind the seam in `src/rhi/` — a future
  Metal port implements the same interfaces without touching `core/`,
  `io/`, `convert/`, or `app/`. Don't bypass the seam to call CUDA/OptiX
  directly from those directories.
- Build: `cmake -B build -G Ninja && cmake --build build`.
