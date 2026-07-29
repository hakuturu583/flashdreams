# Handoff: OmniDreams on a single RTX 3090 (~10 FPS, ~20 GB VRAM)

## Goal

Run the OmniDreams single-view interactive-drive demo on **one 24 GB Ampere GPU
(RTX 3090)**, targeting **~10 effective FPS** while holding the **VRAM peak
around ~20 GB**. Resolution is fixed at **640 x 352 (~360p)** — the resolution
class used by real-time end-to-end driving stacks (openpilot, CARLA E2E
planners), chosen as the best balance of driving-realistic input vs. the
FPS/VRAM budget on a 3090.

## Session to continue from

- Session ID: `session_01SXzW5FW5GQpdLhXeogvccS`
- URL: https://claude.ai/code/session_01SXzW5FW5GQpdLhXeogvccS

## Repo / branch

- Repo: `hakuturu583/flashdreams`
- Branch: `claude/omnidreams-rtx3090-ahncny` (pushed)
- `omni-dreams` repo: **not modified** (all inference/interactive code lives in
  `flashdreams`).

## What has already been changed on this branch

Only two files (no Python logic changed — config + docs only):

1. `integrations/omnidreams/omnidreams/interactive_drive/configs/example_world_model_rtx3090.yaml`
   — new manifest. PyTorch bf16 path (native accel **disabled**), LightVAE +
   LightTAE, `compile_net: true` (torch.compile + CUDA graphs),
   `skip_finalize_kv_cache: true`, `denoising_steps: [1000, 100]`, default
   `resolution_wh: [640, 352]`.
2. `docs/source/models/omnidreams.rst` — added the "Single 24 GB GPU (e.g. RTX
   3090)" section.

Commits:

```
d9b8700  Default RTX 3090 manifest to 640x352 (driving-typical ~360p)
f776d11  Tighten RTX 3090 manifest toward a ~20 GB VRAM budget
711aa28  Add single 24 GB GPU (RTX 3090) interactive-drive manifest
```

## Why this shape (key constraints)

- **No FP8 native path on Ampere.** The published perf path (GB300 numbers)
  needs the OmniDreams single-view native CUDA extension, whose
  `fp8_kvcache_cudnn` / cuDNN / Sparge / SageAttention backends require a
  **Blackwell-class GPU (SM 12.0)**. RTX 3090 is SM 8.6 → must stay on the
  standard PyTorch bf16 path (`native_dit_acceleration: disabled`).
- **OmniDreams minimum is ~48 GB** at the default 720p with the text encoder
  resident. The three Ampere-usable levers are: (1) `--offload-text-encoder`
  (~15 GB saving), (2) reduced resolution, (3) `compile_net: true` for
  steady-state FPS.
- **Model native resolution is 720p (1280 x 704)** — does NOT fit a 3090 even
  with the text encoder offloaded, so native/nuScenes-scale resolutions are off
  the table on this card.

## Setup + run (on the 3090 machine)

```bash
git clone https://github.com/hakuturu583/flashdreams.git
cd flashdreams
git checkout claude/omnidreams-rtx3090-ahncny

# HF token needs read access to nvidia/omni-dreams-models and
# nvidia/omni-dreams-scenes
export HF_TOKEN=<your-hf-token>

uv sync --package flashdreams-omnidreams --extra interactive-drive

# expandable_segments trims reserved-but-unused VRAM (helps hold ~20 GB)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --package flashdreams-omnidreams interactive-drive \
    --manifest example_world_model_rtx3090.yaml \
    --offload-text-encoder \
    --stream-mjpeg :8080
```

Then open `http://<server-ip>:8080/` and pick a scene. First launch is slow
(one-time torch.compile / CUDA-graph capture / Triton autotune); later launches
reuse cached kernels.

Optional pre-staging (avoid first-run network blocking):

```bash
uv run --package flashdreams-omnidreams omnidreams-prepare
```

## What to measure and report back

1. **VRAM peak** via `nvidia-smi` (target: <= ~20 GB). Watch the init peak —
   checkpoint load + first-chunk compile/graph capture is the worst moment.
2. **Effective FPS** (target: ~10). The on-screen HUD / profiler shows per-chunk
   timing; `--profile-world-model` enables CUDA-event profiling.
3. Whether it OOMs, and if so at which stage (load vs. first chunk vs.
   steady state).

Paste the VRAM peak + FPS back into the session so the config can be tuned to
land the targets.

## Tuning ladder (apply in this order if targets are missed)

1. Keep `--offload-text-encoder` on (biggest single saving, ~15 GB).
2. Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
3. If VRAM > ~20 GB or FPS < 10: lower `resolution_wh` in the manifest to
   `[512, 288]` (helps both FPS and VRAM; costs image quality).
4. Last resort for VRAM: set `compile_net: false` — drops the CUDA-graph private
   pool (saves reserved VRAM) but lowers FPS, a direct trade against the 10 FPS
   target.
5. If there is FPS *and* VRAM headroom instead: step `resolution_wh` up toward
   `[896, 496]` / `[1024, 560]` for more fidelity.

Tested, aspect-compatible resolutions (all multiples of 16):
`[1024, 560]`, `[896, 496]`, `[640, 352]` (default), `[512, 288]`.

## Notes / caveats

- FPS and the real VRAM peak are **unverified** — they were designed without
  access to a 3090. Treat 640 x 352 / ~20 GB as a starting point, not a
  guaranteed operating point.
- To continue committing from the 3090 machine, that machine needs push access
  to `hakuturu583/flashdreams` and the same `HF_TOKEN`. Stack further tuning
  commits on the same branch.
- No PR has been opened yet.
