"""Real tiny archives exercise six-view publication and preservation guarantees."""
from pathlib import Path
from unittest.mock import patch
import hashlib,io,json,tarfile,tempfile,unittest
from PIL import Image
from tools import prepare_multicamera as prep

class SixCameraExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.patch=patch.object(prep,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.folder=self.root/'data/bench2drive/extracted/route';(self.folder/'anno').mkdir(parents=True)
        (self.folder/'anno/00000.json.gz').write_bytes(b'unchanged annotation')
        self.payload={}
        for i,c in enumerate(prep.CAMERAS):
            b=io.BytesIO();Image.new('RGB',(10,6),(i*30,0,0)).save(b,format='JPEG');self.payload[c]=b.getvalue()
        front=self.folder/'camera/rgb_front/00000.jpg';front.parent.mkdir(parents=True);front.write_bytes(self.payload['rgb_front'])
    def archive(self,missing_back=False):
        path=self.root/'data/bench2drive/archives/route.tar.gz';path.parent.mkdir(parents=True)
        with tarfile.open(path,'w:gz') as tar:
            for c,data in self.payload.items():
                name='00001.jpg' if missing_back and c=='rgb_back' else '00000.jpg'
                member=tarfile.TarInfo(f'route/camera/{c}/{name}');member.size=len(data);tar.addfile(member,io.BytesIO(data))
            for name in ['route/lidar/00000.bin','route/camera/depth_front/00000.png','route/camera/rgb_top_down/00000.jpg']:
                member=tarfile.TarInfo(name);member.size=4;tar.addfile(member,io.BytesIO(b'data'))
        return dict(path='route.tar.gz',size=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def test_only_six_views_front_unchanged_and_resume(self):
        entry=self.archive();row=prep.audit_episode('bench2drive',entry,True)
        self.assertTrue(row['extracted']);self.assertEqual(len(list((self.folder/'camera').glob('*/*.jpg'))),6)
        self.assertEqual((self.folder/'camera/rgb_front/00000.jpg').read_bytes(),self.payload['rgb_front'])
        self.assertEqual((self.folder/'anno/00000.json.gz').read_bytes(),b'unchanged annotation')
        self.assertFalse((self.folder/'lidar').exists());self.assertFalse((self.folder/'camera/rgb_top_down').exists())
        (self.folder/'camera/rgb_back/00000.jpg').unlink()
        self.assertTrue(prep.audit_episode('bench2drive',entry,True)['extracted'])
    def test_conflicting_front_never_overwritten(self):
        entry=self.archive();front=self.folder/'camera/rgb_front/00000.jpg';front.write_bytes(b'custom front')
        with self.assertRaisesRegex(ValueError,'Conflicting existing RGB'):prep.audit_episode('bench2drive',entry,True)
        self.assertEqual(front.read_bytes(),b'custom front')
    def test_missing_state_linked_view_not_published(self):
        row=prep.audit_episode('bench2drive',self.archive(missing_back=True),True)
        self.assertFalse(row['extracted']);self.assertEqual(row['cameras']['rgb_back']['missing_metadata_ids'],[0])
        self.assertFalse((self.folder/'camera/rgb_front_left').exists())
if __name__=='__main__':unittest.main()
