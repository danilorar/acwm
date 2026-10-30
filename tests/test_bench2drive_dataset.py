"""Synthetic contract tests plus small real-Mini integration checks (unittest)."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.bench2drive_dataset import Bench2DriveDataset, route_group, split_scenarios

PROJECT = Path(__file__).resolve().parents[1]


class DatasetContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.name = 'Example_Town01_Route1_Weather0'
        self.make_clip(self.name)

    def make_clip(self, name, yaw=0., frames=range(8)):
        folder = self.root / name
        (folder / 'anno').mkdir(parents=True)
        (folder / 'camera/rgb_front').mkdir(parents=True)
        for frame in frames:
            angle = np.deg2rad(yaw)
            location = [10 + frame*np.cos(angle), 20 + frame*np.sin(angle), 0.]
            row = dict(speed=float(frame), steer=frame*.01, throttle=frame*.1,
                       brake=float(frame % 2), x=999+frame, y=-999-frame, theta=angle+np.pi/2,
                       bounding_boxes=[{'class':'vehicle'}, {'class':'ego_vehicle',
                           'location':location, 'rotation':[0,0,yaw]}])
            self.save(name, frame, row)
            Image.new('RGB', (12, 8), (frame, 0, 0)).save(folder/'camera/rgb_front'/f'{frame:05}.png')

    def path(self, name, frame):
        return self.root/name/'anno'/f'{frame:05}.json.gz'

    def read(self, name, frame):
        with gzip.open(self.path(name, frame), 'rt') as f:return json.load(f)

    def save(self, name, frame, row):
        with gzip.open(self.path(name, frame), 'wt') as f:json.dump(row,f)

    def dataset(self, **kwargs):
        return Bench2DriveDataset(self.root, history=4, horizon=2, image_size=(10, 6), **kwargs)

    def test_shapes_values_and_batching(self):
        ds=self.dataset();sample=ds[0]
        expected={'past_images':(4,3,6,10), 'past_ego_observations':(4,1),
                  'past_actions':(3,3), 'future_actions':(2,3),
                  'future_images':(2,3,6,10), 'future_motion_targets':(2,5)}
        for key,shape in expected.items():
            self.assertEqual(tuple(sample[key].shape),shape)
            self.assertTrue(torch.isfinite(sample[key]).all())
            self.assertEqual(sample[key].dtype,torch.float32)
        self.assertEqual(sample['motion_target_source'],'simulator')
        batch=next(iter(DataLoader(ds,batch_size=2,num_workers=0)))
        self.assertEqual(tuple(batch['future_actions'].shape),(2,2,3))
        self.assertEqual(tuple(batch['past_actions'].shape),(2,3,3))
        self.assertAlmostEqual(float(sample['past_images'][2,0,0,0]),2/255,places=6)

    def test_both_action_offsets_and_causal_past(self):
        for offset in [0,1]:
            ds=self.dataset(action_offset=offset)
            sample=ds[0]
            recorded=torch.tensor([[self.read(self.name,i)[k] for k in ['steer','throttle','brake']] for i in range(8)])
            torch.testing.assert_close(sample['past_actions'],recorded[offset:offset+3])
            torch.testing.assert_close(sample['future_actions'],recorded[3+offset:5+offset])
            self.assertLessEqual(offset+2,3)  # Latest past label never exceeds cutoff.
            self.assertEqual(ds.action_offset,offset)
        self.assertEqual(self.dataset().action_offset,1)

    def test_preserves_simultaneous_controls(self):
        sample=self.dataset(action_offset=0)[0]
        torch.testing.assert_close(sample['future_actions'][0],torch.tensor([.03,.3,1.]))

    def test_forward_and_right_at_multiple_headings(self):
        for i,yaw in enumerate([0,90,180,-90,37]):
            name=f'Turn_Town02_Route{i+2}_Weather0'
            self.make_clip(name,yaw)
            sample=Bench2DriveDataset(self.root,history=4,horizon=2,image_size=(2,2),scenarios=[name])[0]
            torch.testing.assert_close(sample['future_motion_targets'][:,1:3],torch.tensor([[1.,0.],[2.,0.]]),atol=1e-5,rtol=1e-5)
            # Add +2 local-right to the first future actor position.
            row=self.read(name,4);r=np.deg2rad(yaw)
            row['bounding_boxes'][1]['location'][0]+=-2*np.sin(r)
            row['bounding_boxes'][1]['location'][1]+=2*np.cos(r)
            self.save(name,4,row)
            self.assertAlmostEqual(float(Bench2DriveDataset(self.root,history=4,horizon=2,image_size=(2,2),scenarios=[name])[0]['future_motion_targets'][0,2]),2.,places=5)

    def test_heading_wrap(self):
        a=self.read(self.name,3);b=self.read(self.name,4)
        a['bounding_boxes'][1]['rotation'][2]=179
        b['bounding_boxes'][1]['rotation'][2]=-179
        self.save(self.name,3,a);self.save(self.name,4,b)
        target=self.dataset()[0]['future_motion_targets'][0]
        self.assertAlmostEqual(float(target[3]),np.sin(np.deg2rad(2)),places=6)
        self.assertAlmostEqual(float(target[4]),np.cos(np.deg2rad(2)),places=6)

    def test_privileged_pose_not_used_as_input(self):
        before=self.dataset()[0]
        for frame in range(3,6):
            row=self.read(self.name,frame)
            row['x'],row['y'],row['theta']=1e8,-1e8,123
            row['bounding_boxes'][1]['location'][0]+=frame*20
            if frame>3:row['speed']+=100
            self.save(self.name,frame,row)
        after=self.dataset()[0]
        for key in ['past_images','past_ego_observations','past_actions','future_actions']:
            torch.testing.assert_close(before[key],after[key])
        self.assertFalse(torch.equal(before['future_motion_targets'],after['future_motion_targets']))

    def test_future_changes_do_not_change_history(self):
        for offset in [0,1]:
            before=self.dataset(action_offset=offset)[0]
            for frame in [4,5]:
                row=self.read(self.name,frame);row['throttle']+=.25;row['speed']+=5
                self.save(self.name,frame,row)
            after=self.dataset(action_offset=offset)[0]
            for key in ['past_images','past_ego_observations','past_actions']:
                torch.testing.assert_close(before[key],after[key])

    def test_explicit_noisy_pose_fallback(self):
        for frame in range(8):
            row=self.read(self.name,frame)
            row.update(x=float(frame),y=0.,theta=np.pi/2,bounding_boxes=[])
            self.save(self.name,frame,row)
        with self.assertRaisesRegex(ValueError,'ego_vehicle'):self.dataset()[0]
        sample=self.dataset(pose_source='gps_compass')[0]
        torch.testing.assert_close(sample['future_motion_targets'][:,1:3],torch.tensor([[1.,0.],[2.,0.]]))
        self.assertEqual(sample['motion_target_source'],'gps_compass')

    def test_missing_images_annotations_and_continuity(self):
        (self.root/self.name/'camera/rgb_front/00002.png').unlink()
        self.path(self.name,5).unlink()
        with warnings.catch_warnings(record=True) as caught:
            ds=Bench2DriveDataset(self.root,history=1,horizon=1,image_size=(2,2))
        self.assertTrue(caught)
        self.assertEqual(ds.matching_report[self.name],dict(missing_images=[2],missing_annotations=[5]))
        self.assertEqual([frames for _,frames in ds.samples],[[0,1],[3,4],[6,7]])

    def test_single_history_observation(self):
        for offset in [0,1]:
            sample=Bench2DriveDataset(self.root,history=1,horizon=2,action_offset=offset)[0]
            self.assertEqual(tuple(sample['past_actions'].shape),(0,3))

    def test_invalid_configuration_and_duplicate_ids(self):
        for kwargs in [dict(history=0),dict(horizon=-1),dict(history=True),dict(action_offset=2),dict(action_offset=1.0),dict(image_size=(0,2)),dict(scenarios=['unknown'])]:
            with self.assertRaises(ValueError):Bench2DriveDataset(self.root,**kwargs)
        with self.assertRaisesRegex(ValueError,'No consecutive'):Bench2DriveDataset(self.root,history=100)
        with self.assertRaises(FileNotFoundError):Bench2DriveDataset(self.root/'missing')
        Image.new('RGB',(2,2)).save(self.root/self.name/'camera/rgb_front/0.jpg')
        with self.assertRaisesRegex(ValueError,'Duplicate RGB'):self.dataset()

    def test_nonfinite_rejected(self):
        row=self.read(self.name,5);row['speed']=float('nan');self.save(self.name,5,row)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):self.dataset()[0]

    def test_nested_collection_layout(self):
        name='DifferentScenario/Town03_45'
        self.make_clip(name)
        ds=self.dataset(scenarios=[name])
        self.assertEqual(ds[0]['scenario'],name)
        self.assertEqual(route_group(name),'Town03/Route45')


class SplitContract(unittest.TestCase):
    names=['A_Town01_Route1_Weather0','B_Town01_Route1_Weather2',
           'C_Town01_Route2_Weather0','D_Town02_Route1_Weather0']

    def test_reproducible_route_grouped_split(self):
        a=split_scenarios(self.names,seed=7)
        self.assertEqual(a,split_scenarios(list(reversed(self.names)),seed=7))
        train,val=a
        self.assertFalse(set(train)&set(val))
        self.assertFalse({route_group(n) for n in train}&{route_group(n) for n in val})
        self.assertEqual(set(train+val),set(self.names))
        self.assertEqual(self.names[0] in train,self.names[1] in train)

    def test_explicit_assignments_and_route_conflicts(self):
        train,val=split_scenarios(self.names,val_scenarios=[self.names[2]])
        self.assertEqual(val,[self.names[2]])
        self.assertEqual(split_scenarios(self.names,train_scenarios=train,val_scenarios=val),(train,val))
        for kwargs in [dict(val_scenarios=[self.names[0]]),dict(val_scenarios=['unknown']),
                       dict(train_scenarios=[self.names[0]],val_scenarios=[self.names[2]]),
                       dict(val_scenarios=[]),dict(val_fraction=1)]:
            with self.assertRaises(ValueError):split_scenarios(self.names,**kwargs)

    def test_route_name_aliases_cannot_cross_splits(self):
        names=['A_Town03_Route001_Weather0','B_Town3_Route1_Weather1']
        self.assertEqual(route_group(names[0]),route_group(names[1]))
        with self.assertRaises(ValueError):split_scenarios(names,val_scenarios=[names[0]])

    def test_unknown_route_requires_explicit_provenance(self):
        with self.assertRaisesRegex(ValueError,'route_ids'):split_scenarios(['foo','bar'])
        train,val=split_scenarios(['foo','bar'],route_ids={'foo':'route1','bar':'route2'})
        self.assertEqual(len(train),1);self.assertEqual(len(val),1)
        with self.assertRaises(ValueError):split_scenarios(self.names[:2])


@unittest.skipUnless((PROJECT/'data/bench2drive_mini').is_dir(),'Mini data not installed')
class MiniIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ds=Bench2DriveDataset(PROJECT/'data/bench2drive_mini',image_size=(12,8))

    def test_loading_batching_and_numerics(self):
        self.assertEqual(len(self.ds.scenarios),10)
        self.assertEqual(len(self.ds),2165)
        sample=self.ds[0]
        self.assertEqual(tuple(sample['past_ego_observations'].shape),(4,1))
        self.assertEqual(tuple(sample['future_actions'].shape),(10,3))
        self.assertEqual(tuple(sample['future_motion_targets'].shape),(10,5))
        batch=next(iter(DataLoader(self.ds,batch_size=3,num_workers=0)))
        self.assertEqual(tuple(batch['past_images'].shape),(3,4,3,8,12))
        self.assertEqual(tuple(batch['future_motion_targets'].shape),(3,10,5))
        self.assertTrue(all(torch.isfinite(x).all() for x in batch.values() if isinstance(x,torch.Tensor)))

    def test_all_windows_match_and_remain_consecutive(self):
        for name,frames in self.ds.samples:
            self.assertTrue(all(b==a+1 for a,b in zip(frames,frames[1:])))
            self.assertTrue(set(frames)<=set(self.ds.scenarios[name]['annotations']))
            self.assertTrue(set(frames)<=set(self.ds.scenarios[name]['images']))

    def test_multiple_scenarios_actual_pose_heading_conventions(self):
        heading_bins=set()
        for name,info in self.ds.scenarios.items():
            rows=[self.ds._load_annotation(path) for _,path in sorted(info['annotations'].items())]
            ratios=[];offsets=[]
            for a,b in zip(rows,rows[1:]):
                if a['speed']<=3:continue
                p,yaw=self.ds._pose(a);q,_=self.ds._pose(b)
                dx,dy=q-p;norm=np.hypot(dx,dy)
                if norm<.1:continue
                ratios.append((np.cos(yaw)*dx+np.sin(yaw)*dy)/norm)
                offsets.append(np.angle(np.exp(1j*(a['theta']-yaw))))
                heading_bins.add(round(float(yaw)/(np.pi/2)))
            self.assertGreater(np.median(ratios),.95,name)
            self.assertAlmostEqual(float(np.median(offsets)),np.pi/2,delta=.001)
            # Actual returned target must agree with independent actor locations/yaw.
            index=next(i for i,(n,_) in enumerate(self.ds.samples) if n==name)
            sample=self.ds[index];frames=self.ds.samples[index][1]
            anchor=self.ds._load_annotation(info['annotations'][frames[3]])
            first=self.ds._load_annotation(info['annotations'][frames[4]])
            origin,yaw=self.ds._pose(anchor);destination,_=self.ds._pose(first)
            expected=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])@(destination-origin)
            np.testing.assert_allclose(sample['future_motion_targets'][0,1:3].numpy(),expected,atol=1e-5)
        self.assertGreaterEqual(len(heading_bins),3)

    def test_split_no_frame_or_route_overlap_and_existing_fold_assignments(self):
        train,val=split_scenarios(self.ds.scenarios,seed=2026)
        train_ds=Bench2DriveDataset(self.ds.root,scenarios=train,image_size=(2,2))
        val_ds=Bench2DriveDataset(self.ds.root,scenarios=val,image_size=(2,2))
        frames=lambda ds:{(name,f) for name,window in ds.samples for f in window}
        self.assertFalse(frames(train_ds)&frames(val_ds))
        config=json.loads((PROJECT/'experiments/action_alignment/results/config.json').read_text())
        for held_out in config['validation_folds']:
            _,validation=split_scenarios(self.ds.scenarios,val_scenarios=held_out)
            self.assertEqual(validation,sorted(held_out))

    def test_historical_files_preserved(self):
        manifest=Path('/tmp/bench2drive-refactor-preserved.json')
        if not manifest.exists():self.skipTest('One-time pre-refactor manifest unavailable')
        for name,digest in json.loads(manifest.read_text()).items():
            self.assertEqual(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest(),digest,name)


if __name__=='__main__':unittest.main()
