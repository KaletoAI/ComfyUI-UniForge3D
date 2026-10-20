"""Unofficial ComfyUI bridge to the official UniForge3D multi-view pipeline."""
from pathlib import Path
import subprocess
import tempfile
import uuid

import numpy as np
from PIL import Image

import folder_paths
import comfy.model_management as model_management

from .configuration import runtime_paths


class UniForge3DGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image_front": ("IMAGE",),
                "input_name": ("STRING", {"default": "uniforge3d"}),
                "input_face_num": ("INT", {"default": 100000, "min": 1, "max": 30000000}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
                "input_steps": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "0 keeps upstream step counts; positive values apply to all sampling stages."}),
                "texture_size": ([1024, 2048, 4096], {"default": 2048}),
            },
            "optional": {
                "input_image_back": ("IMAGE",),
                "input_image_left": ("IMAGE",),
                "input_image_right": ("IMAGE",),
                "input_material": (["predicted", "dielectric"], {"default": "predicted"}),
                "input_alpha": (["predicted", "opaque"], {"default": "predicted"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_file",)
    FUNCTION = "generate"
    CATEGORY = "3d/UniForge3D"
    DESCRIPTION = "Unofficial bridge: full upstream 512 → 1024 → PBR, isolated low-VRAM worker. Consistent views of one object."

    def generate(self, input_image_front, input_name, input_face_num, seed,
                 input_steps, texture_size,
                 input_image_back=None, input_image_left=None, input_image_right=None,
                 input_material="predicted", input_alpha="predicted"):
        runtime = runtime_paths()
        root = Path(folder_paths.get_output_directory()) / "uniforge3d" / uuid.uuid4().hex
        directory, filename, counter, _, _ = folder_paths.get_save_image_path(input_name, str(root))
        output = Path(directory) / f"{filename}_{counter:05d}_.glb"
        output.parent.mkdir(parents=True, exist_ok=True)
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        with tempfile.TemporaryDirectory(prefix="uniforge3d-", dir=folder_paths.get_temp_directory()) as directory:
            index = 0
            for views in (input_image_front, input_image_back, input_image_left, input_image_right):
                if views is None:
                    continue
                for tensor in views:
                    pixels = (tensor.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
                    Image.fromarray(pixels).save(Path(directory) / f"{index:03d}.png")
                    index += 1
            if not index:
                raise ValueError("At least one front image is required")
            command = [
                str(runtime.python), str(Path(__file__).with_name("worker.py")),
                "--source", str(runtime.source), "--models", str(runtime.models),
                "--cache", str(runtime.cache), "--input", directory,
                "--output", str(output), "--seed", str(seed),
                "--face-num", str(input_face_num), "--steps", str(input_steps),
                "--texture-size", str(texture_size),
                "--material", input_material, "--alpha", input_alpha,
            ]
            process = subprocess.Popen(command)
            try:
                while True:
                    try:
                        code = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        model_management.throw_exception_if_processing_interrupted()
                if code:
                    raise RuntimeError(f"UniForge3D worker exited with status {code}; see ComfyUI logs")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if not output.is_file():
            raise RuntimeError("UniForge3D worker did not produce a GLB")
        return (str(output),)


NODE_CLASS_MAPPINGS = {"UniForge3DGenerate": UniForge3DGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"UniForge3DGenerate": "UniForge3D Multi-view → Textured GLB"}
