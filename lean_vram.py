"""Bound peak VRAM of the upstream UniForge3D stages without changing results.

Upstream keeps more on the GPU than each step needs; on very dense geometry
(hair, fur, thin strands) that alone exceeds 24 GB. Nothing here changes
weights, precision, resolutions, sampler settings or the order of numeric
operations; only when tensors and models live on the GPU:

- Texture stage: upstream holds the texture denoiser (~5.5 GB FP32) and the VGGT
  conditioner (~3.9 GB) on the GPU while the shape/texture decoders and hole
  filling run. Sampling and decoding now use separate model blocks.
- CuMesh allocates with raw ``cudaMalloc`` and cannot reuse blocks cached by
  PyTorch's allocator; the cache is emptied right before CuMesh-heavy steps.
- Shape-512 postprocess: the raw decoded mesh, the pre-fill CuMesh buffers and
  the BVH stayed alive during narrow-band remeshing; they are released as soon
  as CuMesh owns a copy or the step that needs them is done.
- ``remesh_narrow_band_dc``: same algorithm and values, but dead temporaries are
  freed and the per-voxel edge-neighbour table is built only for intersected
  edges instead of for every voxel (144 bytes per voxel).
"""
import gc

import torch


def release_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gather_edge_neighbours(coords, intersected, edge_neighbor_voxel_offset):
    """Upstream's ``edge_neighbor_voxel[intersected != 0]`` without the full table.

    Upstream: ``(coords.reshape(N, 1, 1, 3) + offset)[mask]`` and
    ``intersected[mask]`` for ``mask = intersected != 0`` of shape ``[N, 3]``.
    Boolean masking and ``nonzero`` both enumerate in row-major order, so the
    gathered rows match upstream in value and order.
    """
    nvox = coords.shape[0]
    if tuple(intersected.shape) != (nvox, 3):
        raise RuntimeError(f"Unexpected dual-contour edge mask shape {tuple(intersected.shape)}")
    hit = (intersected != 0).nonzero()
    voxel, axis = hit[:, 0], hit[:, 1]
    connected_voxel = coords[voxel].reshape(-1, 1, 3) + edge_neighbor_voxel_offset[0][axis]
    return connected_voxel, intersected[voxel, axis]


def lean_remesh_narrow_band_dc(
    vertices,
    faces,
    center,
    scale,
    resolution,
    band=1,
    project_back=0,
    verbose=False,
    bvh=None,
):
    """Memory-lean ``cumesh.remeshing.remesh_narrow_band_dc`` (CuMesh 12289e1).

    Identical operations and outputs; only tensor lifetimes differ. The only
    reformulation is ``connected_voxel``: upstream materializes
    ``coords[:, None, None] + offsets`` for all voxels and then masks it; here
    the same rows are gathered first (``nonzero`` and boolean masking both
    enumerate in row-major order), so values and order are unchanged.
    """
    from cumesh import _C
    from cumesh.bvh import cuBVH
    from cumesh.remeshing import _init_hashmap

    assert vertices.ndim == 2 and vertices.shape[1] == 3 and vertices.dtype == torch.float32
    assert faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype == torch.int32
    assert center.ndim == 1 and center.shape[0] == 3

    device = vertices.device
    fn = lean_remesh_narrow_band_dc
    if not hasattr(fn, "edge_neighbor_voxel_offset"):
        fn.edge_neighbor_voxel_offset = torch.tensor([
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ], dtype=torch.int32).unsqueeze(0).to(device)
    if not hasattr(fn, "quad_split_1_p"):
        fn.quad_split_1_n = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long).to(device)
        fn.quad_split_1_p = torch.tensor([0, 2, 1, 0, 3, 2], dtype=torch.long).to(device)
        fn.quad_split_2_n = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long).to(device)
        fn.quad_split_2_p = torch.tensor([0, 3, 1, 3, 2, 1], dtype=torch.long).to(device)

    offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=torch.int32, device=device)

    if bvh is None:
        bvh = cuBVH(vertices, faces)

    eps = band * scale / resolution
    base_resolution = resolution
    while base_resolution > 32:
        assert base_resolution % 2 == 0, "Failed to find a base resolution that is a multiple of 2"
        base_resolution //= 2

    coords = torch.stack(torch.meshgrid(
        torch.arange(base_resolution, device=device),
        torch.arange(base_resolution, device=device),
        torch.arange(base_resolution, device=device),
        indexing="ij",
    ), dim=-1).int().reshape(-1, 3)

    while True:
        cell_size = scale / base_resolution
        pts = ((coords.float() + 0.5) / base_resolution - 0.5) * scale + center
        distances = bvh.unsigned_distance(pts)[0]
        del pts
        distances -= eps
        distances = torch.abs(distances)
        subdiv_mask = distances < 0.87 * cell_size
        del distances
        coords = coords[subdiv_mask]
        del subdiv_mask
        if base_resolution >= resolution:
            break
        base_resolution *= 2
        coords *= 2
        coords = coords.unsqueeze(1) + offsets.unsqueeze(0)
        coords = coords.reshape(-1, 3)

    nvox = coords.shape[0]
    hashmap_vox = _init_hashmap(resolution, 2 * nvox, device)
    keys = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap_vox, keys, resolution, resolution, resolution)
    del keys

    coords = coords.contiguous()
    grid_verts = _C.get_sparse_voxel_grid_active_vertices(
        *hashmap_vox, coords, resolution, resolution, resolution)
    nvert = grid_verts.shape[0]

    pts_vert = (grid_verts.float() / resolution - 0.5) * scale + center
    distances_vert = bvh.unsigned_distance(pts_vert)[0]
    del pts_vert
    distances_vert -= eps
    if project_back <= 0:
        # Only the projection step below would need the BVH again.
        bvh = None

    hashmap_vert = _init_hashmap(resolution + 1, 2 * nvert, device)
    keys = torch.cat([torch.zeros_like(grid_verts[:, :1]), grid_verts], dim=1)
    _C.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap_vert, keys, resolution + 1, resolution + 1, resolution + 1)
    del keys, grid_verts

    dual_verts, intersected = _C.simple_dual_contour(
        *hashmap_vert, coords, distances_vert, resolution + 1, resolution + 1, resolution + 1
    )
    del hashmap_vert, distances_vert

    connected_voxel, intersected = gather_edge_neighbours(
        coords, intersected, fn.edge_neighbor_voxel_offset)
    del coords
    m = connected_voxel.shape[0]
    lookup = torch.cat([
        torch.zeros((m * 4, 1), dtype=torch.int, device=device),
        connected_voxel.reshape(-1, 3),
    ], dim=1)
    del connected_voxel
    connected_voxel_indices = _C.hashmap_lookup_3d_cuda(
        *hashmap_vox, lookup, resolution, resolution, resolution).reshape(m, 4).int()
    del lookup, hashmap_vox
    connected_voxel_valid = (connected_voxel_indices != 0xffffffff).all(dim=1)
    quad_indices = connected_voxel_indices[connected_voxel_valid].int()
    intersected_dir = intersected[connected_voxel_valid].int()
    del connected_voxel_indices, connected_voxel_valid, intersected

    unique_verts = torch.unique(quad_indices.reshape(-1))
    dual_verts = dual_verts[unique_verts]
    vert_map = torch.zeros((nvox,), dtype=torch.int32, device=device)
    vert_map[unique_verts] = torch.arange(unique_verts.shape[0], dtype=torch.int32, device=device)
    del unique_verts
    quad_indices = vert_map[quad_indices]
    del vert_map

    mesh_vertices = (dual_verts / resolution - 0.5) * scale + center
    del dual_verts
    direction = (intersected_dir == 1).unsqueeze(1)
    del intersected_dir

    def split_alignment(split_p, split_n):
        triangles = torch.where(direction, quad_indices[:, split_p], quad_indices[:, split_n])
        normals0 = torch.cross(
            mesh_vertices[triangles[:, 1]] - mesh_vertices[triangles[:, 0]],
            mesh_vertices[triangles[:, 2]] - mesh_vertices[triangles[:, 0]])
        normals1 = torch.cross(
            mesh_vertices[triangles[:, 2]] - mesh_vertices[triangles[:, 1]],
            mesh_vertices[triangles[:, 3]] - mesh_vertices[triangles[:, 1]])
        alignment = (normals0 * normals1).sum(dim=1).abs()
        return triangles, alignment

    triangles0, align0 = split_alignment(fn.quad_split_1_p, fn.quad_split_1_n)
    triangles1, align1 = split_alignment(fn.quad_split_2_p, fn.quad_split_2_n)
    del quad_indices, direction
    choose_first = (align0 > align1).unsqueeze(1)
    del align0, align1
    mesh_triangles = torch.where(choose_first, triangles0, triangles1).reshape(-1, 3)
    del choose_first, triangles0, triangles1

    if project_back > 0:
        _, face_id, uvw = bvh.unsigned_distance(mesh_vertices, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]]
        projected_verts = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
        mesh_vertices -= project_back * (mesh_vertices - projected_verts)

    return mesh_vertices, mesh_triangles.int()


def lean_shape_only_postprocess(
    raw,
    *,
    grid_size,
    pre_simplify_target,
    decimation_target,
    remesh_band,
    remesh_project,
):
    """Upstream ``official_shape_only_postprocess`` with early releases.

    ``raw`` is a one-element list holding ``(vertices, faces)`` so this function
    can drop the caller's only GPU reference once CuMesh owns a copy. A CPU copy
    is kept solely for upstream's fill_holes fallback path.
    """
    import warnings

    import cumesh
    import trimesh

    vertices, faces = raw.pop()
    device, dtype = vertices.device, vertices.dtype
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    fallback = (vertices.cpu(), faces.cpu())
    del vertices, faces
    release_cuda()
    mesh.simplify(pre_simplify_target, verbose=False)
    try:
        mesh.fill_holes(max_hole_perimeter=3e-2)
    except RuntimeError as error:
        if "[CuMesh]" not in str(error) or "invalid configuration argument" not in str(error):
            raise
        warnings.warn(
            "CuMesh fill_holes rejected this mesh; continuing from the simplified mesh.",
            RuntimeWarning,
        )
        del mesh
        release_cuda()
        mesh = cumesh.CuMesh()
        mesh.init(fallback[0].to(device), fallback[1].to(device))
        mesh.simplify(pre_simplify_target, verbose=False)
    del fallback
    filled_vertices, filled_faces = mesh.read()
    # The filled copy is all the remesh needs; drop CuMesh's internal buffers.
    del mesh
    release_cuda()
    scale = (grid_size + 3 * remesh_band) / grid_size
    remeshed = cumesh.remeshing.remesh_narrow_band_dc(
        filled_vertices,
        filled_faces,
        center=torch.zeros(3, device=device, dtype=dtype),
        scale=scale,
        resolution=grid_size,
        band=remesh_band,
        project_back=remesh_project,
        verbose=False,
        bvh=cumesh.cuBVH(filled_vertices, filled_faces),
    )
    del filled_vertices, filled_faces
    release_cuda()
    mesh = cumesh.CuMesh()
    mesh.init(*remeshed)
    del remeshed
    mesh.simplify(decimation_target, verbose=False)
    mesh.compute_vertex_normals()
    output_vertices, output_faces = mesh.read()
    output_normals = mesh.read_vertex_normals()

    vertices_np = output_vertices.cpu().numpy()
    faces_np = output_faces.cpu().numpy()
    normals_np = output_normals.cpu().numpy()
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    return trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
    )


def lean_decode_shape_and_texture(shape_decoder, texture_decoder, shape_vae_latent, texture_vae_latent):
    """Upstream ``decode_shape_and_texture`` with the cache emptied before CuMesh."""
    meshes, subdivisions = shape_decoder(shape_vae_latent, return_subs=True)
    voxels = texture_decoder(texture_vae_latent, guide_subs=subdivisions) * 0.5 + 0.5
    del subdivisions
    mesh, voxel = meshes[0], voxels[0]
    del meshes, voxels
    release_cuda()
    mesh.fill_holes()
    return mesh, voxel


LAYER_NORM_CHUNK_ROWS = 1 << 18


class _InplaceSiluFunctional:
    """``torch.nn.functional`` for ``sparse_unet``, with ``silu`` applied in place.

    Every ``F.silu`` in upstream ``sparse_unet.py`` (UniForge 8b60947)
    directly follows a ``LayerNorm32`` whose output tensor nothing else holds, so
    writing the activation back saves one full-size buffer per block without
    changing a single value.
    """

    def __getattr__(self, name):
        return getattr(torch.nn.functional, name)

    @staticmethod
    def silu(input, inplace=False):
        return torch.nn.functional.silu(input, inplace=True)


def chunked_layer_norm32(original_forward, chunk_rows=LAYER_NORM_CHUNK_ROWS):
    """Row-chunked ``LayerNorm32.forward`` for large 2-D sparse feature tensors.

    Upstream casts the whole fp16 activation to fp32, normalizes into another
    fp32 tensor and casts back: five fp16-sized buffers at the peak. LayerNorm
    normalizes each row independently, so running the unchanged upstream forward
    on row blocks and writing into one output tensor yields the same values.
    Autocast (where upstream skips the casts) and non-2-D inputs keep upstream.
    """

    def forward(self, x):
        if x.dim() != 2 or x.shape[0] <= chunk_rows or torch.is_autocast_enabled():
            return original_forward(self, x)
        out = torch.empty_like(x)
        for start in range(0, x.shape[0], chunk_rows):
            out[start:start + chunk_rows] = original_forward(self, x[start:start + chunk_rows])
        return out

    return forward


def install_lean_decoder_ops():
    """Patch the sparse VAE decoders' LayerNorm32 and SiLU (values unchanged)."""
    from uniforge3d.models.vaes import sparse_unet
    from uniforge3d.modules import norm

    if not getattr(norm.LayerNorm32.forward, "_lean_vram", False):
        forward = chunked_layer_norm32(norm.LayerNorm32.forward)
        forward._lean_vram = True
        norm.LayerNorm32.forward = forward
    sparse_unet.F = _InplaceSiluFunctional()


def install_lean_cumesh():
    """Route every upstream call of the narrow-band remesher to the lean variant."""
    import cumesh.remeshing

    cumesh.remeshing.remesh_narrow_band_dc = lean_remesh_narrow_band_dc


def lean_vram_pipeline_class(base, pipeline_module, texture_module):
    """Subclass overriding the two stage methods whose GPU residency is too wide."""

    class LeanVramPipeline(base):
        def _generate_shape512(self, batch, structure, dino_features):
            latent, _ = self._sample_shape(
                self.shape512_model,
                self.config["structured_latent_512"],
                batch,
                latent_coords=structure.coords32,
                structure_conditioned_coords=structure.coords64,
                dino_features=dino_features,
                return_view_assignment=False,
            )
            with self._use_models(self.shape512_decoder):
                mesh = pipeline_module.decode_shape_latent(
                    self.shape512_decoder,
                    latent,
                    self.config["structured_latent_512"]["decoder"].get("normalization"),
                )[0]
            del latent
            raw = [(mesh.vertices, mesh.faces)]
            del mesh
            return lean_shape_only_postprocess(raw, **self.config["mesh_postprocess"])

        def _generate_texture(self, batch, normalized_shape_latent, view_assignment, dino_features):
            config = self.config["texture"]
            with self._use_models(self.texture_model, self.conditioner):
                shape_vae_latent = texture_module.denormalize_shape_latent(
                    config, normalized_shape_latent, self.device)
                texture_vae_latent = texture_module.sample_texture(
                    self.texture_model,
                    config,
                    self.conditioner,
                    shape_vae_latent,
                    batch,
                    dino_features,
                    view_assignment,
                    self.device,
                )
            with self._use_models(self.shape1024_decoder, self.texture_decoder):
                reference_mesh, pbr_voxel = lean_decode_shape_and_texture(
                    self.shape1024_decoder, self.texture_decoder, shape_vae_latent, texture_vae_latent)
                del shape_vae_latent, texture_vae_latent
            return texture_module.remesh_and_bake_texture(
                config,
                self.config["mesh_postprocess"],
                reference_mesh,
                pbr_voxel,
            )

    return LeanVramPipeline
