"""Optional export-only material edits; predicted/predicted never rewrites the GLB."""
import argparse
import copy
import io
import json
from pathlib import Path
import struct

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def _read_glb(data):
    if len(data) < 20:
        raise ValueError("Truncated GLB header")
    magic, version, length = struct.unpack_from("<III", data)
    if magic != 0x46546C67 or version != 2 or length != len(data):
        raise ValueError("Expected a complete glTF 2.0 binary file")
    chunks = []
    offset = 12
    while offset < length:
        if offset + 8 > length:
            raise ValueError("Truncated GLB chunk header")
        size, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        if size % 4 or offset + size > length:
            raise ValueError("Invalid GLB chunk length/alignment")
        chunks.append((kind, data[offset:offset + size]))
        offset += size
    if not chunks or chunks[0][0] != JSON_CHUNK:
        raise ValueError("First GLB chunk must be JSON")
    if sum(kind == JSON_CHUNK for kind, _ in chunks) != 1:
        raise ValueError("Expected one JSON chunk")
    if sum(kind == BIN_CHUNK for kind, _ in chunks) > 1:
        raise ValueError("Expected at most one BIN chunk")
    return json.loads(chunks[0][1]), chunks


def _opaque_texture(document, binary, index):
    # Never replace an original image: it may also supply metallic/roughness.
    from PIL import Image

    texture = document["textures"][index]
    extension = texture.get("extensions", {}).get("EXT_texture_webp")
    source = extension["source"] if extension is not None else texture.get("source")
    if source is None:
        raise ValueError("Base-color texture requires an embedded PNG, JPEG or WebP source")
    image = document["images"][source]
    if "uri" in image or "bufferView" not in image:
        raise ValueError("External base-color images are not supported")
    view = document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    end = start + view["byteLength"]
    if view.get("buffer", 0) != 0 or start < 0 or end > len(binary):
        raise ValueError("Base-color image is outside the embedded buffer")
    with Image.open(io.BytesIO(binary[start:end])) as decoded:
        if decoded.format not in {"PNG", "JPEG", "WEBP"} or getattr(decoded, "n_frames", 1) != 1:
            raise ValueError("Expected a single-frame PNG, JPEG or WebP image")
        rgb = decoded.convert("RGB")
        rgb.info.pop("transparency", None)
        encoded = io.BytesIO()
        rgb.save(encoded, format="PNG")
    payload = encoded.getvalue()
    binary.extend(b"\0" * (-len(binary) % 4))
    new_view = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)}
    binary.extend(payload)
    document["bufferViews"].append(new_view)
    document["images"].append({"bufferView": len(document["bufferViews"]) - 1, "mimeType": "image/png"})
    converted = copy.deepcopy(texture)
    converted["source"] = len(document["images"]) - 1
    # A PNG must not inherit an extension that overrides its new core source.
    extensions = converted.get("extensions", {})
    for name in ("EXT_texture_webp", "KHR_texture_basisu", "MSFT_texture_dds"):
        extensions.pop(name, None)
    if not extensions:
        converted.pop("extensions", None)
    document["textures"].append(converted)
    return len(document["textures"]) - 1


def apply_material_options(path, material="predicted", alpha="predicted"):
    """Modify an embedded GLB in place only when explicitly requested.

    Metallic uses the glTF scalar, leaving MR bytes untouched. Opaque appends a
    lossless RGB PNG and a separate texture; old resources and geometry remain.
    RGB equality means equality to the decoded upstream image, not its encoding.
    """
    if material not in {"predicted", "dielectric"} or alpha not in {"predicted", "opaque"}:
        raise ValueError("Unknown material/alpha option")
    if material == "predicted" and alpha == "predicted":
        return
    path = Path(path)
    document, chunks = _read_glb(path.read_bytes())
    binary_index = next((i for i, (kind, _) in enumerate(chunks) if kind == BIN_CHUNK), None)
    binary = None
    converted = {}
    for entry in document.get("materials", []):
        pbr = entry.setdefault("pbrMetallicRoughness", {})
        if material == "dielectric":
            pbr["metallicFactor"] = 0
        if alpha == "opaque" and "baseColorTexture" in pbr:
            if binary_index is None or document.get("buffers", [{}])[0].get("uri") is not None:
                raise ValueError("Opaque conversion requires an embedded GLB buffer")
            if binary is None:
                binary = bytearray(chunks[binary_index][1])
            index = pbr["baseColorTexture"]["index"]
            if index not in converted:
                converted[index] = _opaque_texture(document, binary, index)
            pbr["baseColorTexture"]["index"] = converted[index]
    if binary is not None:
        document["buffers"][0]["byteLength"] = len(binary)
        binary.extend(b"\0" * (-len(binary) % 4))
        chunks[binary_index] = (BIN_CHUNK, bytes(binary))
    encoded = json.dumps(document, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    chunks[0] = (JSON_CHUNK, encoded)
    length = 12 + sum(8 + len(payload) for _, payload in chunks)
    # Serialize fully before opening the destination; validation/decode failures
    # cannot truncate the upstream result.
    result = bytearray(struct.pack("<III", 0x46546C67, 2, length))
    for kind, payload in chunks:
        result.extend(struct.pack("<II", len(payload), kind))
        result.extend(payload)
    path.write_bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glb", type=Path)
    parser.add_argument("--material", choices=("predicted", "dielectric"), default="predicted")
    parser.add_argument("--alpha", choices=("predicted", "opaque"), default="predicted")
    args = parser.parse_args()
    apply_material_options(args.glb, args.material, args.alpha)


if __name__ == "__main__":
    main()
