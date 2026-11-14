"""Click the real Streamlit caption/Img2Img controls with original GPU models.

Only the gallery directory is redirected to temporary storage. Model responses,
SQLite operations, image writes and Streamlit callbacks are not mocked.
"""
import sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'lab7_multimodal_studio'
sys.path.insert(0,str(LAB))
from src.storage.gallery_service import GalleryService
from src.utils.device import ModelLifecycleManager
from streamlit.testing.v1 import AppTest
original_init=GalleryService.__init__
with tempfile.TemporaryDirectory() as directory:
 def isolated_init(self,storage_dir=None,db_name='gallery.db'):
  original_init(self,Path(directory) if storage_dir is None else storage_dir,db_name)
 with patch.object(GalleryService,'__init__',isolated_init):
  ui=AppTest.from_file(str(LAB/'app.py')).run(timeout=120)
  assert not ui.exception and not ui.error,(ui.exception,ui.error)
  ui.slider(key='cap_tokens').set_value(32)
  ui.button(key='btn_gen_cap').click().run(timeout=180)
  assert not ui.exception and not ui.error,(ui.exception,ui.error)
  gallery=GalleryService(Path(directory));records=gallery.get_all_records()
  assert any(r['task_type']=='caption' and r['output_text'] for r in records),records
  print('PASS UI caption -> inference -> SQLite',flush=True)
  ui.slider(key='sd_steps').set_value(15);ui.slider(key='sd_strength').set_value(0.3)
  ui.button(key='btn_sd').click().run(timeout=180)
  assert not ui.exception and not ui.error,(ui.exception,ui.error)
  records=gallery.get_all_records()
  generated=[r for r in records if r['task_type']=='transformation']
  assert generated and (Path(directory)/generated[0]['output_image_path']).is_file(),records
  print('PASS UI Img2Img -> inference -> PNG/SQLite',flush=True)
 ModelLifecycleManager.get_instance().release_all()
print('PASS Streamlit controls and model switching')
