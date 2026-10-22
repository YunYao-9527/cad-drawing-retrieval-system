from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw


def picture(shape):
    image = Image.new('RGB', (100, 100), 'white')
    draw = ImageDraw.Draw(image)
    if shape == 'circle':
        draw.ellipse((15, 15, 85, 85), outline='black', width=5)
    else:
        draw.rectangle((10, 35, 90, 65), fill='black')
    return image


def test_uploaded_image_uses_actual_features_and_stable_order(tmp_path, monkeypatch):
    import demo_main
    monkeypatch.setattr(demo_main, 'GALLERY_DIR', str(tmp_path))
    picture('circle').save(tmp_path / 'circle.png')
    picture('bar').save(tmp_path / 'bar.png')
    demo_main.scan_gallery()
    client = TestClient(demo_main.create_demo_app())
    query = BytesIO()
    picture('circle').save(query, format='PNG')
    first = client.post('/search', files={'image': ('query.png', query.getvalue(), 'image/png')}).json()['results']
    second = client.post('/search', files={'image': ('query.png', query.getvalue(), 'image/png')}).json()['results']
    assert first == second
    assert first[0]['filename'] == 'circle.png'
    assert first[0]['similarity'] == 1.0
    assert first[0]['similarity'] > first[1]['similarity']
    health = client.get('/health').json()
    assert health['clip_loaded'] is False
    assert health['retrieval_method'] == 'ink-density-and-edge-v1'


def test_invalid_upload_and_directory_escape_are_rejected(tmp_path, monkeypatch):
    import demo_main
    monkeypatch.setattr(demo_main, 'GALLERY_DIR', str(tmp_path))
    picture('circle').save(tmp_path / 'circle.png')
    demo_main.scan_gallery()
    client = TestClient(demo_main.create_demo_app())
    assert client.post('/search', files={'image': ('x.png', b'not an image', 'image/png')}).status_code == 400
    assert client.get('/image/%2E%2E%2Foutside.txt').status_code == 400
    monkeypatch.setenv('PUBLIC_DEMO', '1')
    assert client.post('/api/initialize').status_code == 403
    assert client.get('/health').status_code == 200


def test_gallery_query_excludes_itself(tmp_path, monkeypatch):
    import demo_main
    monkeypatch.setattr(demo_main, 'GALLERY_DIR', str(tmp_path))
    picture('circle').save(tmp_path / 'circle.png')
    picture('bar').save(tmp_path / 'bar.png')
    demo_main.scan_gallery()
    client = TestClient(demo_main.create_demo_app())
    result = client.post('/search_by_id', data={'image_id': 'circle.png'}).json()
    assert [row['filename'] for row in result['results']] == ['bar.png']
