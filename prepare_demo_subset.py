"""
从测试集中抽取每类 5 张图片，压缩到 400x300，存入 demo_gallery。
"""
import os
import shutil
import random
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SRC = r"D:\workspace\图纸检索项目\数据集\Dataset_img3"
DST = os.path.join(os.path.dirname(__file__), "demo_gallery")
SAMPLE_PER_CLASS = 5
MAX_SIZE = (400, 300)


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)

    categories = [d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))]
    total = 0

    for cat in sorted(categories):
        src_dir = os.path.join(SRC, cat)
        files = [f for f in os.listdir(src_dir)
                 if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}]
        if not files:
            continue

        sampled = random.sample(files, min(SAMPLE_PER_CLASS, len(files)))
        dst_dir = os.path.join(DST, cat)
        os.makedirs(dst_dir, exist_ok=True)

        for fname in sampled:
            src_path = os.path.join(src_dir, fname)
            dst_name = os.path.splitext(fname)[0] + ".jpg"
            dst_path = os.path.join(dst_dir, dst_name)
            try:
                with Image.open(src_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
                    img.save(dst_path, "JPEG", quality=80, optimize=True)
                total += 1
            except Exception as e:
                print(f"  SKIP {cat}/{fname}: {e}")

    print(f"\nDone: {total} images -> {DST}")


if __name__ == "__main__":
    random.seed(42)
    main()
