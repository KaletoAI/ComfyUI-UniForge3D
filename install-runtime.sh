#!/usr/bin/env bash
# Explicitly opt-in: a NEW isolated worker, never ComfyUI's environment.
set -euo pipefail
if [[ $# != 1 || "$1" != /* ]]; then
    printf 'Usage: bash install-runtime.sh /absolute/new/runtime-directory\n' >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    printf 'This bootstrap supports Linux x86_64 only. See README for other runtimes.\n' >&2
    exit 2
fi
: "${CUDA_HOME:?Set CUDA_HOME to a CUDA 12.4 development toolkit}"
: "${TORCH_CUDA_ARCH_LIST:?Set TORCH_CUDA_ARCH_LIST for your GPU (e.g. 8.6 for RTX 3090)}"
command -v uv >/dev/null
command -v git >/dev/null
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.4'
ROOT="$1"
if [[ -e "$ROOT" || -L "$ROOT" ]]; then
    printf 'Refusing existing path %s; choose a new directory. No environment is modified in place.\n' "$ROOT" >&2
    exit 2
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# mkdir, not mkdir -p: another installer cannot adopt the same destination.
mkdir -- "$ROOT"
clone_pin() {
    local repo="$1" target="$2" revision="$3"
    git init -q "$target"
    git -C "$target" remote add origin "$repo"
    git -C "$target" fetch --depth 1 origin "$revision"
    git -C "$target" checkout --detach FETCH_HEAD
    git -C "$target" submodule update --init --recursive
}
clone_pin https://github.com/xibushu/UniForge3D.git "$ROOT/source" 8b6094780e202b8a6b1d239e0997552e18547026
uv venv --python 3.10 "$ROOT/venv"
PYTHON="$ROOT/venv/bin/python"
uv pip install --python "$PYTHON" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$PYTHON" -c 'import torch; assert torch.version.cuda == "12.4" and not torch._C._GLIBCXX_USE_CXX11_ABI, "FlashAttention wheel ABI mismatch"'
uv pip install --python "$PYTHON" -r "$HERE/requirements-worker.txt"
mkdir "$ROOT/deps"
clone_pin https://github.com/microsoft/TRELLIS.2.git "$ROOT/deps/TRELLIS.2" 75fbf0183001ed9876c8dbb35de6b68552ee08bd
clone_pin https://github.com/JeffreyXiang/CuMesh.git "$ROOT/deps/CuMesh" 12289e1062f0603f2f0d0771b02e1395d247f26f
clone_pin https://github.com/JeffreyXiang/FlexGEMM.git "$ROOT/deps/FlexGEMM" 6dd94a859c26ee8246888502eada3dd8ad85532e
clone_pin https://github.com/NVlabs/nvdiffrast.git "$ROOT/deps/nvdiffrast" 253ac4fcea7de5f396371124af597e6cc957bfae
export CUDA_HOME TORCH_CUDA_ARCH_LIST
export MAX_JOBS="${MAX_JOBS:-2}"
uv pip install --python "$PYTHON" --no-build-isolation --no-deps \
    "$ROOT/deps/nvdiffrast" "$ROOT/deps/CuMesh" "$ROOT/deps/FlexGEMM" "$ROOT/deps/TRELLIS.2/o-voxel"
"$PYTHON" -c 'import torch, flash_attn, cumesh, flex_gemm, o_voxel, nvdiffrast; print("Runtime imports OK:", torch.__version__, flash_attn.__version__)'
printf '\nSet UNIFORGE3D_SOURCE=%s/source and UNIFORGE3D_PYTHON=%s/venv/bin/python\nProvision models separately; see README.\n' "$ROOT" "$ROOT"
