"""Generate placeholder images for the demo gallery."""
from PIL import Image, ImageDraw, ImageFont
import os

GALLERY = os.path.join(os.path.dirname(__file__), "demo_gallery")
CATEGORIES = {
    "轴承": ["深沟球轴承_6205.dwg", "圆锥滚子轴承_30208.dwg", "推力球轴承_51100.dwg"],
    "法兰盘": ["法兰盘_PN16_DN50.dwg", "法兰盘_PN25_DN100.dwg"],
    "齿轮": ["直齿圆柱齿轮_M2_Z30.dwg", "斜齿轮_M3_Z25.dwg", "蜗轮_M4_Z40.dwg"],
    "轴套": ["轴套_D30_L50.dwg", "轴套_D45_L80.dwg"],
    "阀体": ["球阀_DN25.dwg", "闸阀_DN50.dwg"],
}

def make_placeholder(category: str, filename: str, idx: int):
    img = Image.new("RGB", (400, 300), "#f8f9fa")
    draw = ImageDraw.Draw(img)
    # Draw border
    draw.rectangle([2, 2, 397, 297], outline="#3498db", width=2)
    # Draw grid lines
    for x in range(50, 400, 50):
        draw.line([(x, 0), (x, 300)], fill="#e8e8e8", width=1)
    for y in range(50, 300, 50):
        draw.line([(0, y), (400, y)], fill="#e8e8e8", width=1)
    # Draw a simple shape based on category
    cx, cy = 200, 130
    if "轴承" in category:
        draw.ellipse([cx-50, cy-50, cx+50, cy+50], outline="#2c3e50", width=2)
        draw.ellipse([cx-25, cy-25, cx+25, cy+25], outline="#2c3e50", width=2)
    elif "法兰" in category:
        draw.rectangle([cx-50, cy-40, cx+50, cy+40], outline="#2c3e50", width=2)
        draw.ellipse([cx-15, cy-15, cx+15, cy+15], outline="#2c3e50", width=2)
    elif "齿轮" in category:
        draw.ellipse([cx-45, cy-45, cx+45, cy+45], outline="#2c3e50", width=2)
        for angle in range(0, 360, 30):
            import math
            x1 = cx + int(40 * math.cos(math.radians(angle)))
            y1 = cy + int(40 * math.sin(math.radians(angle)))
            x2 = cx + int(55 * math.cos(math.radians(angle)))
            y2 = cy + int(55 * math.sin(math.radians(angle)))
            draw.line([(x1, y1), (x2, y2)], fill="#2c3e50", width=2)
    elif "轴套" in category:
        draw.rectangle([cx-60, cy-20, cx+60, cy+20], outline="#2c3e50", width=2)
        draw.ellipse([cx-60, cy-20, cx-40, cy+20], outline="#2c3e50", width=2)
    else:
        draw.rectangle([cx-35, cy-50, cx+35, cy+50], outline="#2c3e50", width=2)
        draw.rectangle([cx-25, cy-40, cx+25, cy+40], outline="#2c3e50", width=1)
    # Draw text
    draw.text((20, 250), f"[{category}] {filename}", fill="#666")
    draw.text((20, 270), f"Demo #{idx+1}", fill="#999")
    # Save
    fname = os.path.splitext(filename)[0] + ".png"
    path = os.path.join(GALLERY, category, fname)
    img.save(path)
    print(f"  Created: {category}/{fname}")

if __name__ == "__main__":
    idx = 0
    for cat, files in CATEGORIES.items():
        os.makedirs(os.path.join(GALLERY, cat), exist_ok=True)
        for f in files:
            make_placeholder(cat, f, idx)
            idx += 1
    print(f"\nGenerated {idx} demo images in {GALLERY}")
