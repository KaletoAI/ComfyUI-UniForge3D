"""VRAM-bounding overrides must keep upstream values and stage order."""
from contextlib import contextmanager
import types
import unittest

try:
    import torch
except ImportError:  # CPU regressions without torch skip this module.
    torch = None


def upstream_edge_neighbours(coords, intersected, offset):
    nvox = coords.shape[0]
    table = coords.reshape(nvox, 1, 1, 3) + offset
    mask = intersected != 0
    return table[mask], intersected[mask]


@unittest.skipIf(torch is None, "torch not installed")
class EdgeNeighbourGatherTest(unittest.TestCase):
    def test_matches_upstream_value_and_order(self):
        from lean_vram import gather_edge_neighbours

        generator = torch.Generator().manual_seed(3)
        offset = torch.tensor([
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ], dtype=torch.int32).unsqueeze(0)
        coords = torch.randint(0, 1024, (5000, 3), generator=generator, dtype=torch.int32)
        intersected = torch.randint(-1, 2, (5000, 3), generator=generator, dtype=torch.int32)
        expected = upstream_edge_neighbours(coords, intersected, offset)
        actual = gather_edge_neighbours(coords, intersected, offset)
        for want, got in zip(expected, actual):
            self.assertEqual(want.dtype, got.dtype)
            self.assertTrue(torch.equal(want, got))

    def test_rejects_unexpected_mask_shape(self):
        from lean_vram import gather_edge_neighbours

        with self.assertRaises(RuntimeError):
            gather_edge_neighbours(
                torch.zeros((4, 3), dtype=torch.int32),
                torch.zeros((12,), dtype=torch.int32),
                torch.zeros((1, 3, 4, 3), dtype=torch.int32),
            )


@unittest.skipIf(torch is None, "torch not installed")
class TextureStageResidencyTest(unittest.TestCase):
    def setUp(self):
        from lean_vram import lean_vram_pipeline_class

        self.events = []
        events = self.events
        self.active = set()
        active = self.active

        class Mesh:
            def fill_holes(self):
                events.append(("fill_holes", frozenset(active)))

        def decoder(name, output):
            def run(latent, **kwargs):
                events.append((name, frozenset(active)))
                return output
            return run

        class Base:
            config = {"texture": {"t": 1}, "mesh_postprocess": {"m": 2}}
            device = "cpu"
            texture_model, conditioner = "texture_model", "conditioner"
            shape1024_decoder = staticmethod(decoder("shape_decode", ([Mesh()], "subs")))
            texture_decoder = staticmethod(decoder("texture_decode", torch.ones(1, 2)))

            @contextmanager
            def _use_models(self, *models):
                names = {getattr(m, "__name__", m) for m in models}
                events.append(("enter", frozenset(names)))
                active.update(names)
                try:
                    yield
                finally:
                    active.difference_update(names)

        texture = types.SimpleNamespace(
            denormalize_shape_latent=lambda config, latent, device: "shape_vae",
            sample_texture=lambda *args: events.append(("sample", frozenset(active))) or "tex_vae",
            remesh_and_bake_texture=lambda *args: events.append(("bake", frozenset(active))) or args,
        )
        self.pipeline = lean_vram_pipeline_class(Base, types.SimpleNamespace(), texture)()

    def test_samplers_leave_gpu_before_decode(self):
        result = self.pipeline._generate_texture("batch", "latent", {}, "dino")
        by_name = {name: models for name, models in self.events if name != "enter"}
        self.assertIn("texture_model", by_name["sample"])
        self.assertIn("conditioner", by_name["sample"])
        for step in ("shape_decode", "texture_decode", "fill_holes"):
            self.assertNotIn("texture_model", by_name[step])
            self.assertNotIn("conditioner", by_name[step])
        self.assertEqual(by_name["bake"], frozenset())
        self.assertEqual(result[0], {"t": 1})
        self.assertEqual(result[1], {"m": 2})
        self.assertTrue(torch.equal(result[3], torch.ones(2)))
        order = [name for name, _ in self.events if name != "enter"]
        self.assertEqual(order, ["sample", "shape_decode", "texture_decode", "fill_holes", "bake"])


@unittest.skipIf(torch is None, "torch not installed")
class DecoderOpsTest(unittest.TestCase):
    def test_chunked_layer_norm32_matches_upstream(self):
        from lean_vram import chunked_layer_norm32

        class LayerNorm32(torch.nn.LayerNorm):  # upstream uniforge3d.modules.norm
            def forward(self, x):
                x_dtype = x.dtype
                o = super().forward(x.type(torch.float32))
                return o.type(x_dtype)

        generator = torch.Generator().manual_seed(4)
        for affine, dtype in ((True, torch.float16), (False, torch.float16), (True, torch.float32)):
            layer = LayerNorm32(64, elementwise_affine=affine, eps=1e-6)
            if affine:
                with torch.no_grad():
                    layer.weight.copy_(torch.rand(64, generator=generator) + 0.5)
                    layer.bias.copy_(torch.rand(64, generator=generator) - 0.5)
            x = (torch.randn((10007, 64), generator=generator) * 4).to(dtype)
            expected = layer(x)
            actual = chunked_layer_norm32(LayerNorm32.forward, chunk_rows=1000)(layer, x)
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertTrue(torch.equal(expected, actual), (affine, dtype))

    def test_inplace_silu_proxy(self):
        from lean_vram import _InplaceSiluFunctional

        functional = _InplaceSiluFunctional()
        x = torch.randn(1000)
        expected = torch.nn.functional.silu(x.clone())
        target = x.clone()
        result = functional.silu(target)
        self.assertTrue(result is target)
        self.assertTrue(torch.equal(expected, result))
        self.assertIs(functional.layer_norm, torch.nn.functional.layer_norm)


if __name__ == "__main__":
    unittest.main()
