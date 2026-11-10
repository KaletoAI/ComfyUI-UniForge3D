# Changelog

## 1.0.2 (2026-11-10)

Bound peak VRAM of the upstream stages (`lean_vram.py`) so very dense geometry (hair, fur, thin strands) fits a 24 GB RTX 3090. Weights, precision, resolutions, sampler settings and numeric operations stay upstream; only when tensors and models occupy the GPU changes.

- Texture stage: sample with the texture denoiser and VGGT conditioner, move both off the GPU, then decode. Upstream kept about 9 GB of idle sampler weights resident during shape/texture decoding and hole filling.
- Narrow-band remeshing (`cumesh.remeshing.remesh_narrow_band_dc`, used by the Shape-512 postprocess and the PBR bake): same algorithm, dead temporaries freed, edge-neighbour table gathered only for intersected edges. Bit-identical output to CuMesh 12289e1 on a 19 M-face remesh, PyTorch peak 6.35 → 1.84 GiB.
- Shape-512 postprocess: raw decoded mesh, pre-fill CuMesh buffers and BVH released as soon as they are no longer needed; the PyTorch cache is emptied before CuMesh steps, which allocate with raw `cudaMalloc`.
- Sparse VAE decoders: `LayerNorm32` runs in row blocks on large 2-D inputs and `F.silu` after a norm runs in place. Both bit-identical to upstream on an RTX 3090 (fp16/fp32, with/without affine).

Verification: 17 CPU regression tests (five new). RTX 3090, inputs that previously failed with CUDA OOM at Shape-512 (`Kai-hair`) and at shape decoding / hole filling (`Kira-hair`), two views each, dielectric + opaque, defaults otherwise:
- Kira-hair: 375 s, total GPU peak 12.9 GiB, 95,426 triangles, GLB 11.7 MB.
- Kai-hair: 751 s, total GPU peak 23.2 GiB (final shape/texture decode and PBR bake, whose remesh reaches 64 M faces), 95,986 triangles, GLB 7.2 MB. Fits, with little headroom.
- Official four example views, seed 42: total GPU peak 13.8–14.3 GiB → 10.5 GiB, runtime 250 → 259 s.

UniForge3D is not run-to-run reproducible on CUDA: two runs of the previous worker with the same seed produced different decoded meshes (6,379,436 vs 6,324,970 faces). A byte comparison of the full pipeline is therefore impossible; the deterministic pieces above were compared exactly.

## 1.0.1 (2026-11-07)

- Load each upstream stage only while it runs (`lazy_stages.py`). Upstream constructs every stage model (about 30 GB FP32) in CPU RAM up front; `low_vram` only moves the active stage to the GPU. Weights, precision and stage code stay upstream; models are built on first use and dropped after their stage. The shared VGGT conditioner stays loaded across Shape512, Shape1024 and texture.
- Model construction runs inside `torch.random.fork_rng`, so the seeded sampling stream is unchanged by mid-run loading.

Verification: twelve CPU regression tests (incl. four new stage-loading tests; the RNG test fails without `fork_rng`). Real four-view default job on RTX 3090 in a 32 GiB container: 262.9 s, peak anonymous memory 15.6 GiB (previously OOM-killed at 32 GiB), no swap; GLB with 94,965 triangles, 2048² base color and metallic-roughness texture.

## 1.0.0 — 2026-10-20

- Extract the existing ComfyUI UniForge3D process bridge into an independent, explicitly unofficial project.
- Preserve `UniForge3DGenerate`, named view inputs, final-face target, seed and full upstream 512→1024→PBR pipeline.
- Replace host-specific paths and separate worker copies with one configurable worker; enforce offline Hugging Face/Transformers loading before model imports.
- Add optional `input_material=predicted|dielectric` and `input_alpha=predicted|opaque`. Defaults never open/rewrite the exported GLB. Dielectric changes only the metallic scalar. Opaque appends an exact-RGB lossless PNG and preserves original textures, shared MR resources and geometry.
- Include regression coverage for hidden RGB, shared texture roles, multiple materials, PNG transparency metadata, alignment, malformed exports and byte-identical defaults.
- Add opt-in fresh-directory x86_64 CUDA12.4 runtime bootstrap, API workflow, configuration and license documentation. ARM operators supply a separately prepared compatible runtime; no host-specific package symlinks are installed.

Verification: eight CPU regression tests; Python compilation and shell syntax; worker CLI; actual postprocessor CLI on a real UniForge GLB with exact RGB/alpha/MR/geometry checks. Khronos glTF Validator 2.0.0-dev.3.10: original/default/dielectric/opaque/combined each 0 errors and 0 warnings; opaque variants retain an intentionally unused original texture (informational message). New node schema and widgets checked in a disposable ComfyUI 0.37.0 instance and Chromium.

Subsequent production acceptance on RTX 3090: full four-view upstream generation through the gateway with `dielectric` + `opaque`, completed in 283.223 seconds. Downloaded GLB: 8,893,424 bytes, SHA256 `cbe88c3e9b8c11605489ddcda36931b0f3315f460cbc89fcdb50d57af7a35df7`; metallic factor 0, decoded alpha 255, RGB byte-equal to the retained original base-color image. Khronos glTF Validator 2.0.0-dev.3.10: 0 errors, 0 warnings, one informational unused-original-texture message.

This run used the existing CUDA runtime, not a fresh execution of the bootstrap. ARM production cutover remains pending operator service privileges; no new ARM-generation acceptance is claimed.
