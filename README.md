# ComfyUI-UniForge3D

**Unofficial ComfyUI custom node for [UniForge3D](https://github.com/xibushu/UniForge3D): consistent views of one object → textured PBR GLB.** This repository provides the ComfyUI/process bridge, not a new model or an upstream fork.

The complete upstream structure → Shape512 → Shape1024 → PBR pipeline runs in a separate Python process with low-VRAM CPU offloading. ComfyUI's loaded models are unloaded before generation. Cancellation terminates and reaps the worker. Each job has its own output directory and temporary images.

## Requirements and support boundary

- Linux, an NVIDIA CUDA GPU, a working ComfyUI installation with `Preview3D`, and substantial CPU RAM for offloading. The original host integration ran on RTX 3090, RTX 4070 Ti SUPER and ARM64 GB10; these are historical host results, not a fresh acceptance run of every package release or a universal VRAM guarantee.
- A **separate ABI-compatible UniForge3D runtime**, containing Torch, FlashAttention, CuMesh, FlexGEMM, o-voxel and nvdiffrast. Ordinary custom-node installation does not install these into ComfyUI.
- Model weights and complete Hugging Face caches provisioned before generation. Inference is offline; missing files are errors, never an implicit download or runtime fallback.
- No Windows, macOS, CPU-inference or ROCm support is claimed.

## Install the node

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/KaletoAI/ComfyUI-UniForge3D.git
```

There is deliberately **no top-level `requirements.txt`** that makes ComfyUI Manager replace Torch/Transformers in your main environment. The node itself uses ComfyUI's existing NumPy/Pillow. Prepare the separate worker below, configure paths, then restart ComfyUI when idle. Remove any older installation exporting `UniForge3DGenerate` first; do not load two copies.

### Option A: isolated Linux x86_64 worker

The opt-in bootstrap creates a **new, nonexistent directory**. It pins upstream source to `8b6094780e202b8a6b1d239e0997552e18547026` and builds the upstream CUDA extensions. It refuses existing runtime directories rather than modifying them in place.

Prerequisites: `git`, [uv](https://docs.astral.sh/uv/), CUDA **12.4 development toolkit** (`nvcc`, not just the driver), a compatible compiler (the original RTX 3090 recipe used GCC/G++ 13), disk space and network access for source/wheel downloads. Inspect `install-runtime.sh` before executing it. Dependency versions follow upstream; this is not a fully hash-locked environment. The bootstrap is not an automatic ARM installer.

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-UniForge3D
export CUDA_HOME=/path/to/cuda-12.4
export CC=gcc-13 CXX=g++-13
export TORCH_CUDA_ARCH_LIST=8.6  # RTX 3090; choose the architecture for your GPU
bash install-runtime.sh /absolute/new/uniforge-runtime
```

The worker uses Python 3.10, Torch 2.6.0/cu124, torchvision 0.21.0 and the CPython 3.10/Linux x86_64/old-C++-ABI FlashAttention 2.7.3 wheel. Do not use that requirements file in another architecture or Python ABI. Failed installations leave their new directory available for inspection; use another fresh destination to retry.

### Option B: an existing separately prepared runtime (including ARM64)

Point `UNIFORGE3D_PYTHON` at an independently prepared interpreter whose Torch, CUDA extensions and FlashAttention match its GPU/ABI, and `UNIFORGE3D_SOURCE` at the pinned upstream checkout. This package does not copy or symlink binary packages from ComfyUI. Never point it at ComfyUI's own interpreter as a dependency shortcut.

The historical GB10 integration used Python 3.12, Torch 2.12/cu130 and a separately compiled FlashAttention 2.8.3.post2. Its selected binary-package symlinks were host-specific and are **not** a general clean-install recipe. The new node/worker is architecture-neutral; creating an ARM CUDA runtime remains an explicit operator prerequisite.

### Provision models and gated caches

Source MIT licensing does **not** grant unrestricted use of the weights. Read [upstream licensing](https://github.com/xibushu/UniForge3D/tree/8b6094780e202b8a6b1d239e0997552e18547026#license) and accept the relevant gated-repository terms before downloading.

Using the worker interpreter's `huggingface_hub`, download the UniForge model snapshot to `MODELS/uniforge3d`:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "xibushu/UniForge3D",
    revision="a084333f2aae0213e85b9a4541752d1221d618db",
    local_dir="/absolute/models/uniforge3d",
)
```

Then provision all external references from that `pipeline.json` into `/absolute/models/hf-cache/hub` with `hf_hub_download` / `snapshot_download`, preserving each reference's `revision`. Decoder references without extensions require **both** `<filename>.json` and `<filename>.safetensors`. The pipeline also needs DINOv3 and RMBG-2.0 configuration, weights and remote-code files. Gated repositories require your own authorized Hugging Face account. Use `cache_dir="/absolute/models/hf-cache/hub"` for these preparation downloads; the worker will not fetch missing assets.

An existing shared layout also works:

```text
/absolute/models/
├── uniforge3d/
│   ├── pipeline.json
│   └── uniforge3d/…                  # weights from the pinned model snapshot
├── microsoft/…                      # optional local TRELLIS references, repo_id/filename layout
├── facebook/dinov3-vitl16-pretrain-lvd1689m/  # optional local DINOv3 snapshot
└── hf-cache/hub/…                    # complete external/gated model cache
```

Local TRELLIS references are used if present; otherwise upstream resolves them from the offline cache. A local DINOv3 directory overrides its Hub identifier. Other inference configuration and weights are unchanged.

### Configure the ComfyUI process

Set these environment variables **in the process that launches ComfyUI** (for systemd, use a service environment/drop-in rather than only your shell):

| Variable | Meaning |
|---|---|
| `UNIFORGE3D_SOURCE` | Absolute upstream checkout containing `uniforge3d/` |
| `UNIFORGE3D_PYTHON` | Absolute path to the separate runtime's `bin/python` |
| `UNIFORGE3D_MODELS` | Absolute model parent containing `uniforge3d/pipeline.json` |
| `UNIFORGE3D_CACHE` | Optional absolute Hugging Face cache root; default `MODELS/hf-cache` (Hub files in `hub/`) |

```bash
export UNIFORGE3D_SOURCE=/absolute/new/uniforge-runtime/source
export UNIFORGE3D_PYTHON=/absolute/new/uniforge-runtime/venv/bin/python
export UNIFORGE3D_MODELS=/absolute/models
# Start ComfyUI normally after exporting these values.
```

The node registers even before these paths are configured; generation reports missing prerequisites before unloading models. `CUDA_HOME`, `PATH`, `CC` and backend-specific runtime settings remain operator-controlled. The worker enforces `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before importing upstream.

## Node inputs and output

Node ID: **`UniForge3DGenerate`**. Output: **`model_file`**, a `STRING` path to the generated GLB; connect it to ComfyUI `Preview3D` for preview/file output.

| Input | Default | Meaning |
|---|---|---|
| `input_image_front` | required | First/anchor image |
| `input_image_back`, `input_image_left`, `input_image_right` | disconnected | Additional consistent views of the same object |
| `input_name` | `uniforge3d` | Output filename prefix, using ComfyUI's output-path checks |
| `input_face_num` | `100000` | Final target triangle count, not a hard guarantee or Shape512 proxy setting |
| `seed` | `42` | Upstream Torch seed |
| `input_steps` | `0` | Keep upstream stage defaults; positive values set all four samplers to that step count |
| `texture_size` | `2048` | Texture size: 1024, 2048 or 4096 |
| `input_material` | `predicted` | `predicted` or `dielectric` |
| `input_alpha` | `predicted` | `predicted` or `opaque` |

Image batches are additional views of **one object**, not independent generations. The order is front batch, back batch, left batch, right batch. Omit unused views; do not supply unrelated images.

### Material and alpha options

- **`predicted` + `predicted`:** no GLB postprocessing, opening or rewriting. This preserves the upstream file bytes for that run; it is not a promise of cross-hardware GPU determinism.
- **`dielectric`:** set every material's `pbrMetallicRoughness.metallicFactor` to zero. Geometry, roughness and original MR image bytes remain unchanged.
- **`opaque`:** decode each base-color image and append an RGB PNG with **exactly the same decoded RGB values** and implicit alpha 255. Base-color material references are redirected to a separate texture. Original images and textures remain, including `EXT_texture_webp`, so shared MR references are untouched. Upstream `alphaMode` is unchanged (UniForge exports OPAQUE). PNG is lossless but can make the GLB larger; old images remain intentionally.

Use `dielectric` + `opaque` for nonmetallic character exports when desired; retain predicted values for objects where metallic/transparency predictions are useful. These are optional export corrections, not changes to UniForge weights, sampling or geometry.

`workflows/uniforge3d-api.json` is a **ComfyUI API-format** front-only example, not a UI graph export. Upload `front.png`, then POST `{"prompt": <the JSON object>}` to `/prompt`. Add `LoadImage` nodes and the optional view connections for multiview use. API consumers such as AI-Hub must explicitly add `input_material`/`input_alpha` to their own alias mapping/schema; installing the node alone does not update a gateway alias.

## Verify and troubleshoot

CPU-only checks (Pillow with WebP support is needed for the regression fixture):

```bash
python -m pip install Pillow
python -m unittest discover -s tests -v
python -m py_compile __init__.py configuration.py worker.py glb_material.py
bash -n install-runtime.sh
python worker.py --help
```

The GLB postprocessor can also be exercised without CUDA:

```bash
# Operates in place: use a COPY of an original export.
python glb_material.py /path/to/copy.glb --material dielectric --alpha opaque
```

- Node missing: inspect ComfyUI import logs and remove duplicate installations.
- Missing cache/authorization: provision the gated files in the configured cache before retrying; offline mode is deliberate.
- CUDA/extension import error: fix the separate runtime ABI. Do not upgrade ComfyUI packages to hide the failure.
- A full release acceptance additionally needs a real GPU job, ComfyUI preview and a Khronos glTF Validator report. CPU regression tests alone do not prove GPU or preview compatibility.

## License and attribution

This bridge is MIT licensed. The opt-in installation recipe follows [pinned upstream setup.sh](https://github.com/xibushu/UniForge3D/blob/8b6094780e202b8a6b1d239e0997552e18547026/setup.sh); its MIT notice is retained in `LICENSE`. UniForge3D is by its upstream authors; this project is not official or endorsed by them.

No weights, gated code, credentials, private host snapshots or model outputs are bundled. Dependencies and weights retain their own terms: notably VGGT-derived checkpoints are **CC BY-NC 4.0** and **BRIA RMBG-2.0 needs separate commercial authorization**. Publishing an MIT wrapper does not make the pipeline commercially unrestricted.
