#!/usr/bin/env python3
"""Run pinned upstream UniForge3D in a dedicated CUDA-compatible interpreter."""
import argparse
import json
import os
from pathlib import Path
import sys

from glb_material import apply_material_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Hugging Face cache root (containing hub/)")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--face-num", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--texture-size", type=int, choices=(1024, 2048, 4096), default=2048)
    parser.add_argument("--material", choices=("predicted", "dielectric"), default="predicted")
    parser.add_argument("--alpha", choices=("predicted", "opaque"), default="predicted")
    args = parser.parse_args()
    if args.face_num < 1 or not 0 <= args.steps <= 100 or not 0 <= args.seed <= 2147483647:
        parser.error("face-num must be positive; steps 0..100; seed 0..2147483647")
    images = sorted(path for path in args.input.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not images:
        parser.error("No input views")
    os.environ["HF_HOME"] = str(args.cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    sys.path.insert(0, str(args.source))
    from uniforge3d import UniForge3DImageTo3DPipeline

    root = args.models / "uniforge3d"
    config = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    config["texture"]["decimation_target"] = args.face_num
    config["texture"]["texture_size"] = args.texture_size
    if args.steps:
        for stage in ("sparse_structure", "structured_latent_512", "structured_latent_1024", "texture"):
            config[stage]["sampler"]["params"]["steps"] = args.steps

    def local_references(value):
        if isinstance(value, dict):
            if "repo_id" in value and "filename" in value:
                checkpoint = args.models / value["repo_id"] / value["filename"]
                if checkpoint.is_file() or checkpoint.with_suffix(".json").is_file():
                    return str(checkpoint)
            return {key: local_references(item) for key, item in value.items()}
        if isinstance(value, list):
            return [local_references(item) for item in value]
        return value

    config = local_references(config)
    dino = args.models / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"
    if dino.is_dir():
        config["external_models"]["dino"] = str(dino)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    local_config = args.output.parent / "pipeline-local.json"
    local_config.write_text(json.dumps(config), encoding="utf-8")
    pipeline = UniForge3DImageTo3DPipeline.from_config(
        local_config, weights_root=root, low_vram=True,
        cache_dir=str(args.cache / "hub"), local_files_only=True,
    )
    pipeline.run(images, args.output, seed=args.seed)
    apply_material_options(args.output, args.material, args.alpha)
    print(f"UNIFORGE_OUTPUT={args.output}", flush=True)


if __name__ == "__main__":
    main()
