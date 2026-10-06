"""Original-cell grids for the separate V10 artifact recovery; no NN imports."""
import numpy as np
from PIL import Image, ImageDraw


def render_grid(labels, rows):
    if not labels or not rows or any(len(row['images']) != len(labels) for row in rows):
        raise ValueError('Grid needs one exact image per labeled column')
    sheet = Image.new('RGB', (len(labels)*260, 24+len(rows)*288), '#eeeeee')
    draw = ImageDraw.Draw(sheet)
    for j, label in enumerate(labels): draw.text((j*260+2, 3), label, fill='black')
    for i, row in enumerate(rows):
        y = 24+i*288
        for j, rgb in enumerate(row['images']):
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (256,256,3):
                raise ValueError('Use original256 uint8 RGB cells; no implicit scaling')
            draw.text((j*260+2, y+2), row['id'], fill='black')
            sheet.paste(Image.fromarray(rgb), (j*260+2, y+28))
    return sheet
