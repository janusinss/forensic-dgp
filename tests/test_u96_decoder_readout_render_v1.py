"""Meaningful saved-pixel and complete-coverage regressions; no model imports."""
import importlib.util
import io
from pathlib import Path
import unittest

from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / "scripts/render_u96_decoder_readout_return_v1.py"
SPEC = importlib.util.spec_from_file_location("readout_visual_renderer", SOURCE)
RENDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDER)


class RendererTests(unittest.TestCase):
    def test_pixels_preserved_without_resampling(self):
        image = Image.new("RGB", (256,256))
        image.putdata([((x*3+y)%256, (x+y*7)%256, (x*11+y*13)%256)
                       for y in range(256) for x in range(256)])
        stream = io.BytesIO(); image.save(stream, format="PNG")
        decoded = RENDER.rgb256(stream.getvalue())
        sheet = Image.new("RGB", (600,600), "white")
        RENDER.put_cell(sheet, decoded, 21, 48)
        self.assertEqual(image.tobytes(), sheet.crop((21,48,277,304)).tobytes())

    def test_refuses_wrong_size_and_color_mode(self):
        for image in (Image.new("RGB",(255,256)), Image.new("RGBA",(256,256)), Image.new("L",(256,256))):
            stream=io.BytesIO(); image.save(stream,format="PNG")
            with self.assertRaises(ValueError): RENDER.rgb256(stream.getvalue())

    def test_pin_checks_actual_bytes(self):
        data=b"fixed original PNG bytes"
        pin={"bytes": len(data), "sha256": RENDER.digest(data)}
        RENDER.bind_bytes(data,pin)
        with self.assertRaises(ValueError): RENDER.bind_bytes(data+b"x",pin)
        with self.assertRaises(ValueError): RENDER.bind_bytes(b"X"+data[1:],pin)

    def test_primary_coverage_cannot_hide_duplicate_or_missing_output(self):
        pages=[{"rows":[{"generated":[{"record":"a","primary":True},
                                      {"record":"b","primary":True},
                                      {"record":"a","primary":False}]}]}]
        RENDER.validate_coverage(pages,{"a","b"})
        pages[0]["rows"][0]["generated"][1]["record"]="a"
        with self.assertRaises(ValueError): RENDER.validate_coverage(pages,{"a","b"})

    def test_namespace_and_json_reject_ambiguity(self):
        for name in ("../x", "/x", "C:/x", "a\\b", "a//b", "a/./b"):
            with self.assertRaises(ValueError): RENDER.safe_relative(name)
        for value in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError): RENDER.strict_json(value)

    def test_no_torch_checkpoint_or_image_enhancement_apis(self):
        text=SOURCE.read_text()
        for forbidden in ("import torch", "torch.load(", ".resize(", "ImageEnhance", ".extractall(", "pickle.load("):
            self.assertNotIn(forbidden,text)


if __name__ == "__main__": unittest.main()
