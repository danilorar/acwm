"""Offline tests: reproducible selection, authoritative hashes, safe extraction."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from download_bench2drive import select, verified
from extract_bench2drive import safe_member, extract_one


class ExpansionTools(unittest.TestCase):
    def test_balanced_prefix_selection(self):
        entries=[dict(path=f'{name}_{i}',scenario_type=name,town=f'Town{i%2}',route=100*ord(name)+i,weather=i%3) for name in 'ABCD' for i in range(8)]
        first=select(entries,8,42)
        expanded=select(entries,12,42,first)
        self.assertEqual(first,select(list(reversed(entries)),8,42))
        self.assertEqual(expanded,select(entries,12,42))
        self.assertEqual(expanded[:8],first)
        self.assertTrue(all(sum(e['scenario_type']==n for e in expanded)==3 for n in 'ABCD'))

    def test_rejects_traversal_links_devices(self):
        for name in ['/escape','a/../../escape','a\\escape']:
            with self.assertRaises(ValueError):safe_member(tarfile.TarInfo(name))
        for kind in [tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.CHRTYPE]:
            member=tarfile.TarInfo('a/link');member.type=kind
            with self.assertRaises(ValueError):safe_member(member)

    def test_verification_extraction_skip_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'archives').mkdir();(base/'extracted').mkdir()
            archive=base/'archives/test.tar.gz'
            annotation=gzip.compress(json.dumps(dict(speed=1,steer=0,throttle=.3,brake=1,bounding_boxes=[dict(**{'class':'ego_vehicle'},location=[0,0,0],rotation=[0,0,0])])).encode())
            with tarfile.open(archive,'w:gz') as tar:
                for name,payload in [('clip/anno/00000.json.gz',annotation),('clip/camera/rgb_front/00000.jpg',b'raw'),('clip/lidar/unneeded.bin',b'omit')]:
                    member=tarfile.TarInfo(name);member.size=len(payload);tar.addfile(member,io.BytesIO(payload))
            entry=dict(path=archive.name,size=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertTrue(verified(archive,entry))
            record,changed=extract_one(archive,entry,base,{},reserve_bytes=0)
            self.assertTrue(changed);self.assertEqual(len(record['files']),2)
            self.assertFalse((base/'extracted/clip/lidar').exists())
            _,changed=extract_one(archive,entry,base,{entry['path']:record},reserve_bytes=0)
            self.assertFalse(changed)
            archive.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):extract_one(archive,entry,base,{},reserve_bytes=0)


if __name__=='__main__':unittest.main()
