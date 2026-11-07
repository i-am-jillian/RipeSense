"""Offline checks for leakage, feature order, ENVI decoding and notebook preservation."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import ripesense_deephs as dh

class DeepHSTests(unittest.TestCase):
    def test_all_views_days_stay_together(self):
        rows=[]
        for fruit in dh.FRUITS.values():
            for stage in dh.STAGES:
                for number in range(8):
                    for day in ('day_m2_01','day_m2_02'):
                        for side in ('front','back'):
                            specimen=f'{fruit}/m2/{stage}-{number}'
                            rows.append(dict(fruit=fruit,stage=stage,specimen=specimen,day=day,side=side))
        records=pd.DataFrame(rows)
        plan=dh.make_plan(records,'full')
        self.assertTrue((plan.groupby('specimen').split.nunique()==1).all())
        self.assertTrue(set(plan[plan.split=='train'].specimen).isdisjoint(plan[plan.split=='test'].specimen))
        pd.testing.assert_frame_equal(plan,dh.make_plan(records.sample(frac=1,random_state=9),'full'))
        smoke=dh.make_plan(records,'smoke')
        self.assertEqual(set(smoke.fruit),{'mango'})
        self.assertEqual(len(smoke),9)
        self.assertTrue(smoke.specimen.is_unique)

    def test_envi_and_weighted_features(self):
        wavelengths=np.linspace(397.66,1003.81,224)
        cube=np.zeros((32,32,224),dtype='<f4'); cube[5:27,5:27,:]=0.4
        header=b'ENVI\nsamples = 32\nlines = 32\nbands = 224\nheader offset = 0\ndata type = 4\ninterleave = bip\nbyte order = 0\n'
        parsed=dh.read_envi(header,cube.tobytes())
        vector,preview,qc=dh.extract_features(parsed,wavelengths)
        np.testing.assert_allclose(vector,np.full(8,.4),atol=1e-6)
        self.assertEqual(preview.shape,(32,32,3)); self.assertGreater(qc['mask_fraction'],0.1)
        weights=dh.channel_weights(wavelengths)
        np.testing.assert_allclose(weights.sum(axis=0),np.ones(8))
        np.testing.assert_allclose(wavelengths[weights.argmax(axis=0)],dh.CENTERS,atol=2)
        with self.assertRaises(AssertionError):dh.read_envi(header,cube.tobytes()[:-4])
        with self.assertRaises(ValueError):dh.extract_features(np.zeros_like(cube),wavelengths)

    def test_all_fruit_reporting_and_incomplete_batch_gate(self):
        import tempfile
        rows=[]
        for fruit in dh.FRUITS.values():
            for stage_index,stage in enumerate(dh.STAGES):
                for number in range(4):
                    for side in ('front','back'):
                        row=dict(key=f'{fruit}-{stage}-{number}-{side}',fruit=fruit,stage=stage,
                            specimen=f'{fruit}/{stage}/{number}',day='day_m3_01',side=side,
                            split='train' if number<2 else 'test')
                        row.update({name:stage_index+0.01*number for name in dh.FEATURES})
                        rows.append(row)
        features=pd.DataFrame(rows);plan=features.drop(columns=dh.FEATURES)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                dh.train_and_report(plan,features.iloc[:-1],temp,'full')
            bundle,summary,support,pred=dh.train_and_report(plan,features,temp,'full')
            self.assertEqual(set(bundle['models']),set(dh.FRUITS.values()))
            self.assertEqual(bundle['feature_order'],dh.FEATURES)
            self.assertEqual(len(summary),5);self.assertEqual(len(support),15)
            self.assertTrue((support.test_specimens==2).all())
            self.assertTrue((summary.test_specimens==6).all())
            self.assertEqual(len(pred),60)  # two views per held-out specimen
            self.assertTrue((Path(temp)/'deephs_ripeness.joblib').exists())

    def test_notebook_embeds_matching_helper(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'RipeSense_Detection_Spectral.ipynb').read_text())
        cell=next(c for c in nb['cells'] if c['id']=='deephs-04')
        self.assertEqual(cell['source'].strip(),(root/'src/ripesense_deephs.py').read_text().strip())

if __name__=='__main__': unittest.main()
