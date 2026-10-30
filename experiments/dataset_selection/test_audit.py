import io
import pickle
import unittest
import numpy as np
from run_audit import episode, summarize, max_run
from safe_pickle import NumericUnpickler


class AuditInvariants(unittest.TestCase):
    def test_runs_break_on_missing_frames(self):
        self.assertEqual(max_run([True,True,True],[0,1,3]),2)
        self.assertEqual(max_run([True,False,True],[0,1,2]),1)

    def test_one_second_windows_never_bridge_gaps(self):
        ids=np.r_[np.arange(10),np.arange(11,21)]
        e=episode('synthetic','gap',10,ids,np.full(20,3.),np.zeros((20,3)))
        stats=summarize([e])
        self.assertEqual(stats['valid_transitions'],18)
        self.assertEqual(stats['one_second_windows'],0)
        self.assertEqual(stats['max_coast_frames'],10)

    def test_thresholds_and_duration_units(self):
        e=episode('synthetic','run',4,np.arange(9),np.full(9,3.),np.tile([0,0,.5],(9,1)))
        stats=summarize([e])
        self.assertEqual(stats['intermediate_brake_fraction'],1.)
        self.assertEqual(stats['moving_coast_fraction'],0.)
        self.assertEqual(stats['max_intermediate_brake_sampled_seconds'],2.25)
        self.assertEqual(stats['one_second_windows'],5)
        self.assertAlmostEqual(stats['pattern_entropy_bits'],0.)

    def test_coast_requires_moving_and_raw_both_low(self):
        e=episode('synthetic','coast',4,[0,1,2],[0,2,3],[[0,0,0],[0,.01,.01],[0,.02,0]])
        stats=summarize([e]);self.assertEqual(stats['moving_frames'],2)
        self.assertEqual(stats['moving_coast_frames'],1)
        self.assertEqual(stats['moving_coast_fraction'],.5)

    def test_pickle_rejects_arbitrary_globals(self):
        payload=b'cbuiltins\neval\n(V1+1\ntR.'
        with self.assertRaises(pickle.UnpicklingError):NumericUnpickler(io.BytesIO(payload)).load()


if __name__=='__main__':unittest.main()
