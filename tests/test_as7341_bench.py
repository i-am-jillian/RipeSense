import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
import as7341_bench as bench


def row(value,seq=1,**overrides):
    r=dict(seq=seq,millis=seq*700,**dict.fromkeys(bench.CHANNELS,value),gain=16,atime=99,astep=999,led_ma=0,saturated=0)
    r.update(overrides);return r


def line(r): return ','.join(['DATA']+[str(r[k]) for k in bench.FIELDS[1:]])


class FakeModel:
    classes_=np.array(['over-ripe','ripe','unripe'])
    def predict(self,frame):self.input=frame;return ['ripe']
    def predict_proba(self,frame):return [[.1,.8,.1]]


class FakePort:
    def __init__(self,lines):self.lines=iter(lines)
    def reset_input_buffer(self):pass
    def readline(self):return (next(self.lines)+'\n').encode()


class BenchTests(unittest.TestCase):
    def test_channel_order_and_protocol_errors(self):
        r=row(100);r.update({c:100+i for i,c in enumerate(bench.CHANNELS)})
        self.assertEqual(bench.parse_line(line(r)),r)
        self.assertIsNone(bench.parse_line(','.join(bench.FIELDS)))
        for bad in ['ERR,1,200,READ_FAILED','DATA,1,2','DATA,not_an_integer']:
            with self.assertRaises(bench.InvalidScan):bench.parse_line(bad)

    def test_saturation_and_weak_references(self):
        bench.check_raw([row(500),row(600,2)])
        for r in [row(65535),row(50,saturated=1),row(980,atime=0,astep=999)]:
            with self.assertRaises(bench.InvalidScan):bench.check_raw([r])
        dark=bench.summarize([row(10),row(12,2)])
        white=bench.summarize([row(1010),row(1012,2)])
        fruit=bench.summarize([row(510),row(512,2)])
        np.testing.assert_allclose(bench.reference_quality(dark,white),1000)
        np.testing.assert_allclose(bench.normalize(dark,white,fruit),.5)
        with self.assertRaises(bench.InvalidScan):bench.reference_quality(dark,dark)
        noisy={**white,'std':np.full(8,200)}
        with self.assertRaises(bench.InvalidScan):bench.reference_quality(dark,noisy)
        with self.assertRaises(bench.CalibrationChanged):bench.check_raw([row(10),row(10,2,gain=32)])

    def test_saved_order_scores_and_unsupported_apple(self):
        model=FakeModel();order=list(bench.FEATURE_TO_CHANNEL)[::-1]
        bundle={'feature_order':order,'models':{'mango':model}}
        values=np.arange(8)/10
        result=bench.predict(bundle,'mango',values)
        self.assertEqual(result['predicted_stage'],'ripe')
        self.assertEqual(result['class_scores'],{'over-ripe':.1,'ripe':.8,'unripe':.1})
        self.assertEqual(list(model.input.columns),order)
        np.testing.assert_allclose(model.input.iloc[0].to_numpy(),values[::-1])
        self.assertFalse(bench.predict(bundle,'apple',values)['supported'])

    def test_retry_keeps_invalid_raw_and_normalized_data(self):
        class Reader:
            def __init__(self):self.calls=0
            def collect(self,*args):
                self.calls+=1
                return [row(65535 if self.calls==1 else 510),row(65535 if self.calls==1 else 510,2)]
        with tempfile.TemporaryDirectory() as temp:
            log=bench.Log(Path(temp)/'run')
            dark=bench.summarize([row(10),row(10,2)]);white=bench.summarize([row(1010),row(1010,2)])
            with patch('builtins.input',return_value=''),patch('builtins.print'):
                _,values=bench.measure(Reader(),log,'fruit_1','test',2,dark,white)
            log.close();events=[json.loads(x) for x in (Path(temp)/'run/measurements.jsonl').read_text().splitlines()]
            self.assertFalse(events[0]['valid']);self.assertGreater(events[0]['relative_reflectance'][0],1)
            self.assertTrue(events[1]['valid']);np.testing.assert_allclose(values,.5)

    def test_reader_discards_stale_rows_and_detects_board_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            log=bench.Log(Path(temp)/'run')
            port=FakePort([line(row(100,i)) for i in [1,2,3,4,1]])
            reader=bench.Reader(port,log)
            self.assertEqual([r['seq'] for r in reader.collect('dark',2)],[3,4])
            with self.assertRaises(bench.CalibrationChanged):reader.collect('white',2)
            log.close()

    def test_complete_apple_session_logs_three_normalized_scans(self):
        class StreamingPort:
            def __init__(self):self.value=10;self.seq=0
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def reset_input_buffer(self):pass
            def readline(self):
                self.seq+=1
                return (line(row(self.value,self.seq))+'\n').encode()
        port=StreamingPort();fruit_scans=[]
        def prompt(text):
            if text.startswith('DARK'):port.value=10
            elif text.startswith('WHITE'):port.value=1010
            else:
                fruit_scans.append(1);port.value=410+100*(len(fruit_scans)-1)
            return ''
        bundle={'feature_order':list(bench.FEATURE_TO_CHANNEL),'models':{'mango':FakeModel()}}
        with tempfile.TemporaryDirectory() as temp:
            model=Path(temp)/'fixture.joblib';model.write_bytes(b'test fixture')
            with patch.object(bench,'load_bundle',return_value=bundle),patch.object(bench.serial,'Serial',return_value=port),patch.object(bench.time,'sleep'),patch('builtins.input',side_effect=prompt),patch('builtins.print'):
                status=bench.main(['--port','fake','--model',str(model),'--fruit','apple','--samples','2','--log-dir',str(Path(temp)/'logs')])
            self.assertEqual(status,0)
            directory=next((Path(temp)/'logs').iterdir())
            events=[json.loads(x) for x in (directory/'measurements.jsonl').read_text().splitlines()]
            predictions=[x for x in events if x['phase']=='prediction']
            self.assertEqual(len(predictions),3)
            self.assertTrue(all(not p['supported'] for p in predictions))
            np.testing.assert_allclose([p['relative_reflectance'][0] for p in predictions],[.4,.5,.6])
            self.assertEqual(len((directory/'serial.jsonl').read_text().splitlines()),20)
            table=pd.read_csv(directory/'repeatability.csv');np.testing.assert_allclose(table['std'],.1)

    def test_repeatability_sample_sd(self):
        table=bench.repeatability_table([np.full(8,.4),np.full(8,.5),np.full(8,.6)])
        np.testing.assert_allclose(table['mean'],.5);np.testing.assert_allclose(table['std'],.1)
        np.testing.assert_allclose(table['cv_percent'],20)


if __name__=='__main__':unittest.main()
