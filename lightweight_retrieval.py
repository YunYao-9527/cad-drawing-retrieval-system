"""Deterministic image-layout retrieval for memory-constrained demo servers.

This descriptor is not CLIP and its scores are not the formal experiment metrics.
"""
from functools import lru_cache
import math

from PIL import Image, ImageOps


def descriptor(image):
    gray = ImageOps.autocontrast(ImageOps.grayscale(image)).resize((32, 32))
    pixels = list(gray.getdata())
    density = [1 - value / 255 for value in pixels]
    norm = math.sqrt(sum(value * value for value in density)) or 1
    density = tuple(value / norm for value in density)
    edges = tuple(pixels[y * 32 + x] > pixels[y * 32 + x + 1]
                  for y in range(32) for x in range(31))
    return density, edges


@lru_cache(maxsize=512)
def file_descriptor(path, version):
    with Image.open(path) as image:
        return descriptor(image)


def rank_images(image, catalog, top_k=10, exclude_id=None):
    import os
    query_density, query_edges = descriptor(image)
    matches = []
    for item in catalog:
        if item['id'] == exclude_id:
            continue
        try:
            density, edges = file_descriptor(item['filepath'], os.stat(item['filepath']).st_mtime_ns)
        except (OSError, ValueError):
            continue
        cosine = sum(a * b for a, b in zip(query_density, density))
        edge_agreement = sum(a == b for a, b in zip(query_edges, edges)) / len(query_edges)
        similarity = min(1.0, max(0.0, 0.65 * cosine + 0.35 * edge_agreement))
        matches.append({**item, '_similarity': similarity})
    matches.sort(key=lambda item: (-item['_similarity'], item['id']))
    return matches[:min(max(int(top_k), 1), 50)]
