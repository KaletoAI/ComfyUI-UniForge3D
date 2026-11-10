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
        )
        import lean_vram

        def bake(texture_config, mesh_config, inputs):
            events.append(("bake", frozenset(active)))
            return texture_config, mesh_config, inputs.pop("mesh"), inputs.pop("pbr_voxel"), inputs

        original_bake = lean_vram.lean_remesh_and_bake_texture
        lean_vram.lean_remesh_and_bake_texture = bake
        self.addCleanup(setattr, lean_vram, "lean_remesh_and_bake_texture", original_bake)
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
        self.assertEqual(result[4], {})
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


@unittest.skipIf(torch is None, "torch not installed")
class MeshExtractionTest(unittest.TestCase):
    """Lean ``flexible_dual_grid_to_mesh`` vs. o_voxel's inference path (CUDA hashmap faked on CPU)."""

    class FakeVoxelModule:
        @staticmethod
        def _init_hashmap(grid_size, capacity, device):
            return [{}]

        class _C:
            @staticmethod
            def hashmap_insert_3d_idx_as_val_cuda(table, keys, *grid):
                for i, key in enumerate(keys.tolist()):
                    table[tuple(key)] = i

            @staticmethod
            def hashmap_lookup_3d_cuda(table, keys, *grid):
                return torch.tensor([table.get(tuple(k), 0xffffffff) for k in keys.tolist()], dtype=torch.int64)

    @classmethod
    def upstream(cls, coords, dual_vertices, intersected_flag, split_weight, aabb, voxel_size=None, grid_size=None, train=False):
        # o_voxel 0.0.1 convert/flexible_dual_grid.py, inference path (train=False, split_weight given)
        import numpy as np
        vm = cls.FakeVoxelModule
        offset = torch.tensor([
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ], dtype=torch.int).unsqueeze(0)
        aabb = torch.tensor(np.array(aabb), dtype=torch.float32)
        grid_size = torch.tensor(np.array([grid_size] * 3), dtype=torch.int32)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
        N = dual_vertices.shape[0]
        mesh_vertices = (coords.float() + dual_vertices) / (2 * N) - 0.5
        hashmap = vm._init_hashmap(grid_size, 2 * N, device="cpu")
        vm._C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1), *grid_size.tolist())
        edge_neighbor_voxel = coords.reshape(N, 1, 1, 3) + offset
        connected_voxel = edge_neighbor_voxel[intersected_flag]
        M = connected_voxel.shape[0]
        key = torch.cat([torch.zeros((M * 4, 1), dtype=torch.int), connected_voxel.reshape(-1, 3)], dim=1)
        idx = vm._C.hashmap_lookup_3d_cuda(*hashmap, key, *grid_size.tolist()).reshape(M, 4).int()
        valid = (idx != 0xffffffff).all(dim=1)
        quad_indices = idx[valid].int()
        mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)
        sw = split_weight[quad_indices]
        mesh_triangles = torch.where(
            sw[:, 0] * sw[:, 2] > sw[:, 1] * sw[:, 3],
            quad_indices[:, torch.tensor([0, 1, 2, 0, 2, 3])],
            quad_indices[:, torch.tensor([0, 1, 3, 3, 1, 2])],
        ).reshape(-1, 3)
        return mesh_vertices, mesh_triangles

    def test_matches_upstream_inference_path(self):
        from lean_vram import lean_flexible_dual_grid_to_mesh

        generator = torch.Generator().manual_seed(11)
        grid = torch.stack(torch.meshgrid(*[torch.arange(12)] * 3, indexing="ij"), -1).reshape(-1, 3)
        coords = grid[torch.rand(len(grid), generator=generator) < 0.6].int()
        n = coords.shape[0]
        dual = torch.rand((n, 3), generator=generator)
        intersected = torch.rand((n, 3), generator=generator) > 0.3
        split = torch.rand((n, 1), generator=generator)
        kwargs = dict(aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], grid_size=12, train=False)
        lean = lean_flexible_dual_grid_to_mesh(self.upstream, self.FakeVoxelModule)
        want = self.upstream(coords, dual, intersected, split, **kwargs)
        got = lean(coords, dual, intersected, split, **kwargs)
        self.assertGreater(want[1].shape[0], 100)
        for w, g in zip(want, got):
            self.assertEqual(w.dtype, g.dtype)
            self.assertTrue(torch.equal(w, g))

    def test_other_call_forms_go_upstream(self):
        from lean_vram import lean_flexible_dual_grid_to_mesh

        calls = []
        lean = lean_flexible_dual_grid_to_mesh(lambda *a, **k: calls.append(k) or "upstream", self.FakeVoxelModule)
        z = torch.zeros((1, 3))
        self.assertEqual(lean(z.int(), z, z.bool(), None, aabb=[[0] * 3, [1] * 3], grid_size=4), "upstream")
        self.assertEqual(lean(z.int(), z, z.bool(), z[:, :1], aabb=[[0] * 3, [1] * 3], grid_size=4, train=True), "upstream")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
