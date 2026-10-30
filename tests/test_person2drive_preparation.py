"""Exercise selective extraction and resumption using tiny local archives."""
import hashlib,io,sys,tarfile,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import zstandard
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import prepare_person2drive as preparation

class PersonExtraction(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.base=self.root/'person'
  self.patches=[patch.object(preparation,'BASE',self.base),patch.object(preparation,'PROJECT',self.root)]
  for p in self.patches:p.start();self.addCleanup(p.stop)
 def archive(self,members):
  buffer=io.BytesIO()
  with tarfile.open(fileobj=buffer,mode='w') as tar:
   for name,data in members:
    info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
  content=zstandard.ZstdCompressor().compress(buffer.getvalue());path=self.base/'archives/driver/test.tar.zst';path.parent.mkdir(parents=True);path.write_bytes(content)
  index=self.root/'index.npz';np.savez(index,image_paths=np.array(['test/camera/rgb_front/00001.jpg']))
  return dict(path='driver/test.tar.zst',driver='driver',episode='test',index='index.npz',size=len(content),sha256=hashlib.sha256(content).hexdigest())
 def test_front_only_and_idempotent_resume(self):
  entry=self.archive([('test/camera/rgb_front/00001.jpg',b'front'),('test/lidar/00001.bin',b'large unused modality'),('test/camera/rgb_back/00001.jpg',b'back')])
  preparation.extract(entry,{});preparation.extract(entry,{})
  files=[p.relative_to(self.base/'extracted/driver/test').as_posix() for p in (self.base/'extracted/driver/test').rglob('*') if p.is_file()]
  self.assertEqual(files,['camera/rgb_front/00001.jpg']);self.assertTrue((self.base/'archives'/entry['path']).exists())
 def test_unsafe_member_never_publishes_episode(self):
  entry=self.archive([('../escape',b'bad'),('test/camera/rgb_front/00001.jpg',b'front')])
  with self.assertRaisesRegex(ValueError,'Unsafe'):preparation.extract(entry,{})
  self.assertFalse((self.base/'extracted/driver/test').exists());self.assertFalse((self.root/'escape').exists())
 def test_missing_indexed_image_never_publishes_episode(self):
  entry=self.archive([('test/camera/rgb_front/00002.jpg',b'wrongframe')])
  with self.assertRaisesRegex(ValueError,'Missing referenced'):preparation.extract(entry,{})
  self.assertFalse((self.base/'extracted/driver/test').exists())
if __name__=='__main__':unittest.main()
