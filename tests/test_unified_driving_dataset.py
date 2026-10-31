import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from src.unified_driving_dataset import UnifiedDrivingDataset,CAMERA_ORDER,resize_camera_intrinsics


class UnifiedContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def record(self,source,sign,yaw=0):
        angle=np.deg2rad(yaw);forward=np.array([np.cos(angle),np.sin(angle),0.])
        positions=np.arange(8)[:,None]*forward
        camera_paths={c:[] for c in CAMERA_ORDER}
        for ci,c in enumerate(CAMERA_ORDER):
            for i in range(8):
                path=self.root/f'{source}_{c}_{i}.png';Image.new('RGB',(10,6),(ci*30,i,0)).save(path);camera_paths[c].append(path)
        paths=camera_paths['rgb_front']
        return dict(source=source,episode='test',driver='human' if sign<0 else '',route='1',
                    frame_ids=np.arange(8),speed=np.arange(8,dtype=np.float32),
                    actions=np.array([[i*.01,i*.1,float(i%2)] for i in range(8)],np.float32),
                    positions=positions,forward=np.tile(forward,(8,1)),right_sign=sign,image_paths=paths,camera_paths=camera_paths)

    def test_axes_and_heading_both_handednesses(self):
        for source,sign in [('bench2drive',1),('person2drive',-1)]:
            for yaw in [0,90,-90,179]:
                r=self.record(source,sign,yaw)
                angle=np.deg2rad(yaw);right=sign*np.array([-np.sin(angle),np.cos(angle),0])
                r['positions'][4]+=2*right
                turn=angle+sign*np.deg2rad(10)
                r['forward'][4]=[np.cos(turn),np.sin(turn),0]
                sample=UnifiedDrivingDataset({'ep':r},action_offsets={source:1},horizon=2).numeric_sample(0)
                target=sample['future_motion_targets'][0]
                torch.testing.assert_close(target[1:3],torch.tensor([1.,2.]),atol=1e-5,rtol=1e-5)
                self.assertAlmostEqual(float(target[3]),np.sin(np.deg2rad(10)),places=5)

    def test_both_offsets_and_source_independence(self):
        b=self.record('bench2drive',1);p=self.record('person2drive',-1)
        ds=UnifiedDrivingDataset({'b':b,'p':p},action_offsets={'bench2drive':1,'person2drive':0},horizon=2)
        for index,key,offset in [(0,'b',1),(3,'p',0)]:
            sample=ds.numeric_sample(index);r=ds.records[key]
            torch.testing.assert_close(sample['past_actions'],torch.tensor(r['actions'][offset:offset+3]))
            torch.testing.assert_close(sample['future_actions'],torch.tensor(r['actions'][3+offset:5+offset]))
            self.assertLessEqual(offset+2,3)

    def test_no_privileged_future_leakage(self):
        r=self.record('person2drive',-1);ds=UnifiedDrivingDataset({'p':r},action_offsets={'person2drive':1},horizon=2)
        before=ds.numeric_sample(0);r['positions'][4:]+=100;r['speed'][4:]+=50;r['actions'][4:]+=.1
        after=ds.numeric_sample(0)
        for key in ['past_ego_observations','past_actions']:torch.testing.assert_close(before[key],after[key])
        self.assertFalse(torch.equal(before['future_motion_targets'],after['future_motion_targets']))

    def test_gaps_and_batching(self):
        r=self.record('bench2drive',1);ds=UnifiedDrivingDataset({'b':r},action_offsets={'bench2drive':0},horizon=2,image_size=(8,4))
        batch=next(iter(DataLoader(ds,batch_size=2)))
        self.assertEqual(tuple(batch['past_images'].shape),(2,4,6,3,4,8));self.assertEqual(tuple(batch['future_motion_targets'].shape),(2,2,5))
        self.assertTrue(torch.isfinite(batch['future_motion_targets']).all())
        r['frame_ids'][3:]+=1
        with self.assertRaisesRegex(ValueError,'No contiguous'):UnifiedDrivingDataset({'b':r},action_offsets={'bench2drive':0},horizon=2)

    def test_uncertain_controls_are_preserved_but_default_windows_exclude_them(self):
        r=self.record('person2drive',-1);r['actions'][0,0]=1.8
        strict=UnifiedDrivingDataset({'p':r},action_offsets={'person2drive':0},horizon=2)
        raw=UnifiedDrivingDataset({'p':r},action_offsets={'person2drive':0},horizon=2,exclude_uncertain_controls=False)
        self.assertEqual(len(strict),2);self.assertEqual(strict.samples[0][1],1)
        self.assertEqual(len(raw),3);sample=raw.numeric_sample(0)
        self.assertAlmostEqual(float(sample['past_actions'][0,0]),1.8,places=5)
        self.assertFalse(bool(sample['action_in_nominal_range'][0]))
        self.assertAlmostEqual(float(r['actions'][0,0]),1.8,places=5)

    def test_camera_order_selection_and_missing_views(self):
        r=self.record('bench2drive',1)
        ds=UnifiedDrivingDataset({'b':r},action_offsets={'bench2drive':0},horizon=2,image_size=(10,6))
        sample=ds[0];self.assertEqual(sample['camera_ids'],CAMERA_ORDER)
        for i in range(6):self.assertAlmostEqual(float(sample['past_images'][0,i,0,0,0]),i*30/255,places=5)
        r['camera_paths']['rgb_back'][0].unlink()
        missing=UnifiedDrivingDataset({'b':r},action_offsets={'bench2drive':0},horizon=2)
        self.assertEqual(len(missing),2);self.assertEqual(missing.rejected_windows_missing_cameras,1)
        front=UnifiedDrivingDataset({'b':r},action_offsets={'bench2drive':0},horizon=2,cameras=('rgb_front',))
        self.assertEqual(len(front),3);self.assertEqual(tuple(front[0]['past_images'].shape),(4,1,3,224,224))
        self.assertFalse(front[0]['camera_geometry_valid'].any())
        with self.assertRaises(ValueError):UnifiedDrivingDataset({'b':r},action_offsets={'bench2drive':0},horizon=2,cameras=('bad',))

    def test_resize_geometry_projects_identically(self):
        K=np.array([[100.,0,5],[0,80.,3],[0,0,1]])
        new,A=resize_camera_intrinsics(K,(10,6),(224,224))
        point=np.array([1.,2.,10.]);uv=K@point;uv/=uv[2]
        projected=new@point;projected/=projected[2]
        np.testing.assert_allclose(projected,A@uv)
        self.assertAlmostEqual(new[0,0],2240.)
        cropped,B=resize_camera_intrinsics(K,(10,6),(224,224),(2,1,9,6))
        projected=cropped@point;projected/=projected[2];np.testing.assert_allclose(projected,B@uv)
        with self.assertRaises(ValueError):resize_camera_intrinsics(K,(10,6),(224,224),(-1,0,9,6))

    def test_calibration_filter_and_camera_geometry_batch(self):
        r=self.record('person2drive',-1)
        K=np.tile(np.array([[100.,0,5],[0,80.,3],[0,0,1]]),(8,6,1,1))
        T=np.tile(np.eye(4),(8,6,1,1))
        r['camera_geometry']=dict(intrinsics=K,camera_to_ego=T,native_camera_to_ego=T,valid=np.ones((8,6),bool))
        six=UnifiedDrivingDataset({'p':r},action_offsets={'person2drive':1},horizon=2,require_calibration=True)
        front=UnifiedDrivingDataset({'p':r},action_offsets={'person2drive':1},horizon=2,cameras=('rgb_front',),require_calibration=True)
        a,b=six[0],front[0]
        for key in ['past_actions','future_actions','past_ego_observations','future_motion_targets']:torch.testing.assert_close(a[key],b[key])
        self.assertEqual(tuple(a['camera_intrinsics'].shape),(6,6,3,3))
        torch.testing.assert_close(a['camera_intrinsics'][0,0],a['image_affine'][0,0]@a['camera_intrinsics_original'][0,0])
        r['camera_geometry']['valid'][0,1]=False
        rejected=UnifiedDrivingDataset({'p':r},action_offsets={'person2drive':1},horizon=2,require_calibration=True)
        self.assertEqual(len(rejected),2);self.assertEqual(rejected.rejected_windows_missing_calibration,1)

    def test_source_optical_conversions(self):
        from tools.build_camera_geometry import ACTOR_TO_OPTICAL,LEFT_TO_RIGHT
        bench_camera_to_body=np.linalg.inv(ACTOR_TO_OPTICAL)
        person_camera_to_native=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]],float)
        np.testing.assert_array_equal(bench_camera_to_body,LEFT_TO_RIGHT@person_camera_to_native)
        np.testing.assert_array_equal(bench_camera_to_body@np.array([1,0,0,0]),[0,1,0,0])

    def test_explicit_alignment_required(self):
        r=self.record('person2drive',-1)
        with self.assertRaises(ValueError):UnifiedDrivingDataset({'p':r},action_offsets={},horizon=2)


if __name__=='__main__':unittest.main()
