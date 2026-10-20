"""Guard exported material edits against silent geometry/texture corruption."""
import copy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

from PIL import Image

from glb_material import apply_material_options


def pack(document, binary):
    encoded = json.dumps(document).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    return (struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary))
            + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


def unpack(data):
    size, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A
    document = json.loads(data[20:20 + size])
    length, kind = struct.unpack_from("<II", data, 20 + size)
    assert kind == 0x004E4942
    return document, data[28 + size:28 + size + length]


def image_bytes(document, binary, index):
    view = document["bufferViews"][document["images"][index]["bufferView"]]
    offset = view.get("byteOffset", 0)
    return binary[offset:offset + view["byteLength"]]


def base_image(document, binary, material=0):
    texture = document["textures"][document["materials"][material]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
    source = texture.get("extensions", {}).get("EXT_texture_webp", {}).get("source", texture.get("source"))
    return Image.open(io.BytesIO(image_bytes(document, binary, source))).convert("RGBA")


def fixture():
    image = Image.new("RGBA", (2, 2))
    image.putdata([(111, 27, 9, 0), (3, 150, 33, 80), (19, 47, 203, 180), (91, 20, 77, 255)])
    stream = io.BytesIO()
    image.save(stream, format="WEBP", lossless=True, exact=True)
    texture = stream.getvalue()
    # An ordinary opaque triangle, with a deliberately shared base-color/MR image.
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    normals = struct.pack("<9f", *([0, 0, 1] * 3))
    uv = struct.pack("<6f", 0, 0, 1, 0, 0, 1)
    binary = positions + normals + uv + texture
    document = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["EXT_texture_webp"],
        "extensionsRequired": ["EXT_texture_webp"],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
            {"buffer": 0, "byteOffset": 36, "byteLength": 36, "target": 34962},
            {"buffer": 0, "byteOffset": 72, "byteLength": 24, "target": 34962},
            {"buffer": 0, "byteOffset": 96, "byteLength": len(texture)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
        ],
        "images": [{"bufferView": 3, "mimeType": "image/webp"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9729}],
        "textures": [{"extensions": {"EXT_texture_webp": {"source": 0}}, "sampler": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0, "texCoord": 0}, "metallicRoughnessTexture": {"index": 0}, "metallicFactor": 1, "roughnessFactor": 0.8}, "alphaMode": "OPAQUE"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "material": 0}]}],
        "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0,
    }
    return pack(document, binary)


class MaterialOptionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "mesh.glb"
        self.original = fixture()
        self.path.write_bytes(self.original)
        self.document, self.binary = unpack(self.original)

    def test_predicted_does_not_rewrite_file(self):
        """Default output must remain exactly the upstream file, not a reserialization."""
        before = self.path.stat().st_mtime_ns
        apply_material_options(self.path)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(self.path.stat().st_mtime_ns, before)

    def test_dielectric_preserves_binary_and_roughness(self):
        """Metal correction must not reencode MR or alter roughness/geometry."""
        apply_material_options(self.path, material="dielectric")
        document, binary = unpack(self.path.read_bytes())
        self.assertEqual(binary, self.binary)
        expected = copy.deepcopy(self.document)
        expected["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] = 0
        self.assertEqual(document, expected)

    def test_opaque_preserves_hidden_rgb_and_shared_mr(self):
        """Opaque conversion cannot destroy transparent RGB or a shared MR source."""
        decoded = base_image(self.document, self.binary)
        apply_material_options(self.path, alpha="opaque")
        document, binary = unpack(self.path.read_bytes())
        result = base_image(document, binary)
        self.assertEqual(result.convert("RGB").tobytes(), decoded.convert("RGB").tobytes())
        self.assertEqual(set(result.getchannel("A").tobytes()), {255})
        self.assertEqual(binary[:len(self.binary)], self.binary)
        self.assertEqual(document["images"][:1], self.document["images"])
        self.assertEqual(document["textures"][:1], self.document["textures"])
        self.assertEqual(document["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"], {"index": 0})
        self.assertEqual(image_bytes(document, binary, 0), image_bytes(self.document, self.binary, 0))
        for key in ("meshes", "accessors", "nodes", "scenes", "bufferViews"):
            if key == "bufferViews":
                self.assertEqual(document[key][:len(self.document[key])], self.document[key])
            else:
                self.assertEqual(document[key], self.document[key])
        for view in document["bufferViews"]:
            self.assertEqual(view.get("byteOffset", 0) % 4, 0)
            self.assertLessEqual(view.get("byteOffset", 0) + view["byteLength"], document["buffers"][0]["byteLength"])
        self.assertEqual(len(self.path.read_bytes()) % 4, 0)

    def test_combined_options_update_all_materials_and_keep_binding(self):
        """Multiple materials sharing one base texture retain their own roughness and UV binding."""
        second = copy.deepcopy(self.document["materials"][0])
        second["pbrMetallicRoughness"]["roughnessFactor"] = 0.3
        self.document["materials"].append(second)
        self.path.write_bytes(pack(self.document, self.binary))
        apply_material_options(self.path, material="dielectric", alpha="opaque")
        document, binary = unpack(self.path.read_bytes())
        for material, roughness in zip(document["materials"], (0.8, 0.3)):
            pbr = material["pbrMetallicRoughness"]
            self.assertEqual(pbr["metallicFactor"], 0)
            self.assertEqual(pbr["roughnessFactor"], roughness)
            self.assertEqual(pbr["baseColorTexture"]["texCoord"], 0)
            self.assertEqual(material["alphaMode"], "OPAQUE")
        self.assertEqual(document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"], document["materials"][1]["pbrMetallicRoughness"]["baseColorTexture"]["index"])
        self.assertEqual(set(base_image(document, binary).getchannel("A").tobytes()), {255})

    def test_png_source_without_webp_extension(self):
        """PNG-only exporters must not require EXT_texture_webp to perform an alpha correction."""
        stream = io.BytesIO()
        Image.new("RGBA", (1, 1), (13, 29, 71, 0)).save(stream, format="PNG")
        payload = stream.getvalue()
        self.document.pop("extensionsRequired")
        self.document.pop("extensionsUsed")
        self.document["textures"][0] = {"source": 0}
        self.document["images"][0]["mimeType"] = "image/png"
        self.document["bufferViews"][3]["byteLength"] = len(payload)
        binary = self.binary[:96] + payload
        self.document["buffers"][0]["byteLength"] = len(binary)
        self.path.write_bytes(pack(self.document, binary))
        apply_material_options(self.path, alpha="opaque")
        document, binary = unpack(self.path.read_bytes())
        self.assertEqual(base_image(document, binary).getpixel((0, 0)), (13, 29, 71, 255))

    def test_png_transparent_color_key_is_removed(self):
        """RGB/palette PNG transparency metadata must not survive opaque conversion."""
        for mode, transparency in (("RGB", (13, 29, 71)), ("P", 0)):
            with self.subTest(mode=mode):
                image = Image.new(mode, (1, 1), (13, 29, 71) if mode == "RGB" else 0)
                if mode == "P":
                    image.putpalette([13, 29, 71] + [0] * 765)
                stream = io.BytesIO()
                image.save(stream, format="PNG", transparency=transparency)
                payload = stream.getvalue()
                document = copy.deepcopy(self.document)
                document.pop("extensionsRequired")
                document.pop("extensionsUsed")
                document["textures"][0] = {"source": 0}
                document["images"][0]["mimeType"] = "image/png"
                document["bufferViews"][3]["byteLength"] = len(payload)
                binary = self.binary[:96] + payload
                document["buffers"][0]["byteLength"] = len(binary)
                self.path.write_bytes(pack(document, binary))
                apply_material_options(self.path, alpha="opaque")
                result, binary = unpack(self.path.read_bytes())
                self.assertEqual(base_image(result, binary).getpixel((0, 0)), (13, 29, 71, 255))

    def test_refuses_truncated_glb_without_overwriting(self):
        """A corrupt export must not be replaced by a partially processed GLB."""
        truncated = self.original[:-4]
        self.path.write_bytes(truncated)
        with self.assertRaises(ValueError):
            apply_material_options(self.path, material="dielectric")
        self.assertEqual(self.path.read_bytes(), truncated)

    def test_invalid_option_does_not_silently_use_predicted(self):
        """CLI/API typos must not silently produce an uncorrected material."""
        with self.assertRaises(ValueError):
            apply_material_options(self.path, alpha="opqaue")
        self.assertEqual(self.path.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
