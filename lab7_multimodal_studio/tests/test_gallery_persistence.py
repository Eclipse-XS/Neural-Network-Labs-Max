"""Real SQLite/PNG persistence across Python processes, including Windows handle cleanup."""
import os,subprocess,sys,tempfile
from pathlib import Path
from PIL import Image

LAB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB))
from src.storage.gallery_service import GalleryService

def test_gallery_survives_process_restart_and_releases_handles():
    with tempfile.TemporaryDirectory() as directory:
        gallery=GalleryService(Path(directory))
        image=Image.open(LAB/'assets/01_coastal_lighthouse.png').convert('RGB')
        identifiers=[gallery.add_record('caption','persistence',{},output_text='lighthouse',input_image=image,output_image=image) for _ in range(30)]
        assert len(set(identifiers))==30
        code='''import sys
from pathlib import Path
from src.storage.gallery_service import GalleryService
from PIL import Image
storage=Path(sys.argv[1]);gallery=GalleryService(storage)
assert len(gallery.get_all_records(100))==30
record=gallery.get_record(sys.argv[2]);assert record['output_text']=='lighthouse'
for field in ['input_image_path','output_image_path']:
    with Image.open(storage/record[field]) as image:assert image.size==(512,512)
'''
        subprocess.run([sys.executable,'-c',code,directory,identifiers[-1]],cwd=LAB,check=True,timeout=30)
    # TemporaryDirectory cleanup above fails on Windows if any SQLite handle leaks.
    assert not Path(directory).exists()
