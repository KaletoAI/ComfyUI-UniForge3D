"""Stage-wise loading must bound resident models without changing seeded sampling."""
from contextlib import contextmanager
import types
import unittest

try:
    import torch
except ImportError:  # CPU regressions without torch skip this module.
    torch = None


class FakeModel(torch.nn.Module if torch else object):
    def __init__(self, name, log):
        super().__init__()
        log.append(name)
        # Construction consumes the global RNG like real initializers do.
        self.weight = torch.nn.Parameter(torch.randn(4))


def fake_pipeline_module(log, used):
    class Base:
        def __init__(self, *args, **kwargs):
            raise AssertionError("eager upstream constructor must not run")

        @classmethod
        def from_config(cls, config, **kwargs):
            return cls(config, **kwargs)

        @contextmanager
        def _use_models(self, *models):
            used.append(tuple(models))
            yield

        def run(self):
            torch.manual_seed(7)
            first = torch.randn(3)
            with self._use_models(self.structure_model):
                pass
            second = torch.randn(3)
            with self._use_models(self.shape512_model, self.conditioner):
                pass
            with self._use_models(self.texture_model, self.conditioner):
                pass
            return first, second

    return types.SimpleNamespace(
        UniForge3DImageTo3DPipeline=Base,
        BackgroundRemover=lambda *args, **kwargs: "remover",
        release_cuda=lambda: None,
    )


@unittest.skipIf(torch is None, "torch not installed")
class LazyStagesTest(unittest.TestCase):
    def setUp(self):
        from lazy_stages import STAGE_MODELS, lazy_pipeline_class

        self.log, self.used = [], []
        builders = {name: (lambda name=name: FakeModel(name, self.log)) for name in STAGE_MODELS}
        module = fake_pipeline_module(self.log, self.used)
        cls = lazy_pipeline_class(module, lambda config, **kwargs: builders)
        self.pipeline = cls.from_config(
            {"image_preprocess": {"rembg_model": "x"}}, device=0, low_vram=True,
        )

    def test_constructor_loads_nothing(self):
        self.assertEqual(self.log, [])
        self.assertEqual(self.pipeline.remove_background, "remover")

    def test_stage_models_dropped_after_use_conditioner_kept(self):
        self.pipeline.run()
        self.assertEqual(self.log, ["structure_model", "shape512_model", "conditioner", "texture_model"])
        self.assertEqual(set(self.pipeline._loaded), {"conditioner"})

    def test_lazy_loading_keeps_seeded_rng_stream(self):
        first, second = self.pipeline.run()
        torch.manual_seed(7)
        self.assertTrue(torch.equal(first, torch.randn(3)))
        self.assertTrue(torch.equal(second, torch.randn(3)))

    def test_unknown_attribute_still_raises(self):
        with self.assertRaises(AttributeError):
            self.pipeline.not_a_stage


if __name__ == "__main__":
    unittest.main()
