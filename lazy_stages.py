"""Load each upstream UniForge3D stage only while it runs.

Upstream builds every stage model (about 30 GB of FP32 weights) into CPU RAM at
construction; ``low_vram`` only moves the active stage to the GPU and back. This
subclass keeps the upstream stage code, weights and precision unchanged, but
builds a model on first access and drops it once its ``_use_models`` block ends.
Peak RAM becomes the largest stage instead of the whole pipeline.

Model construction consumes the global RNG (random initialisation before the
checkpoint is loaded). Upstream does that before ``run`` seeds the generator;
here it happens mid-run, so every build runs inside ``torch.random.fork_rng``
to leave the seeded sampling stream exactly as upstream sees it.
"""
from contextlib import contextmanager

STAGE_MODELS = (
    "dino", "conditioner",
    "structure_model", "structure_decoder",
    "shape512_model", "shape512_decoder",
    "shape1024_model", "shape1024_decoder",
    "texture_model", "texture_decoder",
)
# Shared by Shape512, Shape1024 and texture, which run back to back.
STICKY_MODELS = frozenset({"conditioner"})


def upstream_builders(config, *, cache_dir=None, local_files_only=False):
    """Per-model constructors mirroring upstream ``loaders.load_models`` exactly."""
    from easydict import EasyDict as edict
    from uniforge3d.models import loaders

    structure = edict(config["sparse_structure"])
    shape512 = edict(config["structured_latent_512"])
    shape1024 = edict(config["structured_latent_1024"])
    texture = edict(config["texture"])

    def dino():
        return loaders.DinoV3Encoder(
            config["external_models"]["dino"],
            cache_dir=cache_dir, local_files_only=local_files_only,
        ).eval().requires_grad_(False)

    def structure_model():
        spec = structure.model
        model = loaders.get_model_class(spec.name)(**spec.args)
        loaders._load_structure_weights(model, structure.weights)
        return model

    return {
        "dino": dino,
        "conditioner": lambda: loaders.StructuredLatentConditioner(**config["conditioner"]),
        "structure_model": structure_model,
        "structure_decoder": lambda: loaders.load_sparse_structure_decoder(structure.decoder.checkpoint),
        "shape512_model": lambda: loaders._load_flow_model(shape512.model, shape512.weights),
        "shape512_decoder": lambda: loaders.load_shape_decoder(
            shape512.decoder.checkpoint, int(shape512.decoder.resolution)),
        "shape1024_model": lambda: loaders._load_flow_model(shape1024.model, shape1024.weights),
        "shape1024_decoder": lambda: loaders.load_shape_decoder(
            texture.shape_decoder_checkpoint, int(texture.resolution)),
        "texture_model": lambda: loaders._load_flow_model(texture.model, texture.weights),
        "texture_decoder": lambda: loaders.load_texture_decoder(texture.texture_decoder_checkpoint),
    }


def lazy_pipeline_class(pipeline_module, builders_factory=upstream_builders):
    import torch

    base = pipeline_module.UniForge3DImageTo3DPipeline

    class LazyStagePipeline(base):
        def __init__(self, config, *, device=0, low_vram=True, cache_dir=None, local_files_only=False):
            # Deliberately skips base.__init__, which loads every stage eagerly.
            self.config = config
            self.device = torch.device("cuda", int(device))
            self.low_vram = low_vram
            self._builders = builders_factory(
                config, cache_dir=cache_dir, local_files_only=local_files_only)
            self._loaded = {}
            with torch.device("cpu"):
                self.remove_background = pipeline_module.BackgroundRemover(
                    config["image_preprocess"]["rembg_model"], self.device,
                    cache_dir=cache_dir, local_files_only=local_files_only,
                )

        def __getattr__(self, name):
            if name.startswith("_") or name not in STAGE_MODELS:
                raise AttributeError(name)
            model = self._loaded.get(name)
            if model is None:
                with torch.random.fork_rng(devices=[]), torch.device("cpu"):
                    model = self._builders[name]()
                self._loaded[name] = model
            return model

        @contextmanager
        def _use_models(self, *models):
            with super()._use_models(*models):
                yield
            for name, model in list(self._loaded.items()):
                if name not in STICKY_MODELS and any(model is used for used in models):
                    del self._loaded[name]
            # Upstream already moved the models back to CPU and released CUDA;
            # the dropped ones are freed once the caller's references end.

    return LazyStagePipeline
