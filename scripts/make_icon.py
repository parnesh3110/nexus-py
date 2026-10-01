"""Draws the Nexus app icon (the Jarvis orb on a dark rounded square) at every size macOS
wants, and packs it into an .icns file.   python scripts/make_icon.py out/Nexus.icns"""
import subprocess
import sys
import tempfile
from pathlib import Path

from AppKit import (NSBezierPath, NSBitmapImageRep, NSCalibratedRGBColorSpace, NSColor, NSGradient,
                    NSGraphicsContext)

PALETTE = [(1.00, 0.33, 0.62), (0.62, 0.36, 1.00), (0.24, 0.55, 1.00), (0.20, 0.86, 0.93)]
PNG = 4  # NSBitmapImageFileTypePNG


def draw(size: int) -> bytes:
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, NSCalibratedRGBColorSpace, 0, 0)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    s = float(size)
    tile = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(((s * 0.1, s * 0.1), (s * 0.8, s * 0.8)), s * 0.18, s * 0.18)
    NSGradient.alloc().initWithStartingColor_endingColor_(
        NSColor.colorWithSRGBRed_green_blue_alpha_(0.12, 0.12, 0.17, 1),
        NSColor.colorWithSRGBRed_green_blue_alpha_(0.04, 0.04, 0.07, 1)).drawInBezierPath_angle_(tile, -90)
    orb = NSBezierPath.bezierPathWithOvalInRect_(((s * 0.27, s * 0.27), (s * 0.46, s * 0.46)))
    NSGradient.alloc().initWithColors_([NSColor.colorWithSRGBRed_green_blue_alpha_(*c, 1) for c in PALETTE]
                                       ).drawInBezierPath_angle_(orb, 45)
    NSGradient.alloc().initWithStartingColor_endingColor_(
        NSColor.colorWithWhite_alpha_(1.0, 0.85), NSColor.colorWithWhite_alpha_(1.0, 0.0)
    ).drawInBezierPath_relativeCenterPosition_(orb, (-0.35, 0.35))
    NSGraphicsContext.restoreGraphicsState()
    return bytes(rep.representationUsingType_properties_(PNG, {}))


def main(out: str):
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "Nexus.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            (iconset / f"icon_{base}x{base}.png").write_bytes(draw(base))
            (iconset / f"icon_{base}x{base}@2x.png").write_bytes(draw(base * 2))
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", out], check=True)
    print(f"icon: {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Nexus.icns")
