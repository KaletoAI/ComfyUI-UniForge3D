# Changelog

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
