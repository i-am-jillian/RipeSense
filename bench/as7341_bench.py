#!/usr/bin/env python3
"""Guided local AS7341 bench test. No camera, Colab or training required."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import serial
from serial.tools import list_ports
from sklearn.exceptions import InconsistentVersionWarning

CHANNELS = [f'F{i}' for i in range(1, 9)]
WAVELENGTHS = [415,445,480,515,555,590,630,680]
FEATURE_TO_CHANNEL = {f'{c}_{nm}nm_mean_reflectance':c for c,nm in zip(CHANNELS,WAVELENGTHS)}
FIELDS = ['kind','seq','millis',*CHANNELS,'gain','atime','astep','led_ma','saturated']
SETTINGS = ['gain','atime','astep','led_ma']


class InvalidScan(ValueError):
    pass


class CalibrationChanged(RuntimeError):
    pass


def parse_line(line):
    parts=next(csv.reader([line.strip()]))
    if not parts or parts[0].startswith('#') or parts[0]=='kind': return None
    if parts[0]=='ERR': raise InvalidScan('Arduino read error: '+','.join(parts[1:]))
    if len(parts)!=len(FIELDS) or parts[0]!='DATA': raise InvalidScan('Malformed serial row: '+line.strip())
    try: row={k:int(v) for k,v in zip(FIELDS[1:],parts[1:])}
    except ValueError as e: raise InvalidScan('Non-integer serial field') from e
    if any(not 0<=row[c]<=65535 for c in CHANNELS): raise InvalidScan('ADC value outside uint16 range')
    if (row['gain'] not in [1,2,4,8,16,32,64,128,256,512] or
        not 0<=row['atime']<=255 or not 0<=row['astep']<=65534 or
        not 0<=row['led_ma']<=258 or row['saturated'] not in (0,1) or row['seq']<0):
        raise InvalidScan('Invalid hardware settings/status')
    return row


def load_bundle(path):
    # Load only the user's trusted training artifact, never an arbitrary downloaded pickle.
    with warnings.catch_warnings():
        warnings.simplefilter('error',InconsistentVersionWarning)
        try: bundle=joblib.load(path)
        except InconsistentVersionWarning as e:
            raise ValueError('Model/library version mismatch. Install bench/requirements.txt (supplied model: sklearn 1.6.1).') from e
    order=bundle.get('feature_order',[])
    if len(order)!=8 or set(order)!=set(FEATURE_TO_CHANNEL):
        raise ValueError('Unsupported feature schema; expected the eight DeepHS F1–F8 mean-reflectance features')
    if not isinstance(bundle.get('models'),dict): raise ValueError('Bundle must contain models dict')
    for fruit,model in bundle['models'].items():
        if getattr(model,'n_features_in_',None)!=8: raise ValueError(f'{fruit}: expected eight inputs')
        if list(getattr(model,'feature_names_in_',[]))!=list(order):
            raise ValueError(f'{fruit}: pipeline feature names differ from bundle order')
    return bundle


def predict(bundle,fruit,reflectance):
    if fruit not in bundle['models']:
        return {'fruit':fruit,'supported':False,'message':f'No {fruit} ripeness model in this bundle; sensor-only scan.'}
    order=bundle['feature_order']
    row={feature:float(reflectance[CHANNELS.index(FEATURE_TO_CHANNEL[feature])]) for feature in order}
    frame=pd.DataFrame([row],columns=order)
    model=bundle['models'][fruit]
    # Fitted StandardScaler and classifier are already in the pipeline. No extra scaling/refit.
    stage=str(model.predict(frame)[0]); scores=model.predict_proba(frame)[0]
    return {'fruit':fruit,'supported':True,'predicted_stage':stage,
            'class_scores':{str(k):float(v) for k,v in zip(model.classes_,scores)},
            'experimental_live_sensor':True,'sensor_calibrated_accuracy_established':False}


def summarize(rows):
    a=np.array([[r[c] for c in CHANNELS] for r in rows],dtype=float)
    return {'mean':a.mean(axis=0),'std':a.std(axis=0,ddof=1),'min':a.min(axis=0),'max':a.max(axis=0)}


def check_raw(rows):
    configs={tuple(r[k] for k in SETTINGS) for r in rows}
    if len(configs)!=1: raise CalibrationChanged('Hardware settings changed: restart dark/white calibration.')
    r=rows[0]; ceiling=min(65535,(r['atime']+1)*(r['astep']+1))
    if any(r['saturated'] or any(r[c]>=0.98*ceiling for c in CHANNELS) for r in rows):
        raise InvalidScan('Saturation/near saturation. Reduce fixed light or gain, then restart ALL calibration; do not reuse references.')


def reference_quality(dark,white,min_counts=100,snr=10):
    denominator=white['mean']-dark['mean']
    noise=np.sqrt(dark['std']**2+white['std']**2)
    required=np.maximum(min_counts,snr*noise)
    weak=denominator<=required
    if np.any(weak):
        raise InvalidScan('Weak/nonpositive white-dark reference in '+', '.join(np.array(CHANNELS)[weak])+
                          f'; need >max({min_counts:g} counts, {snr:g} × observed noise). Repeat white or restart calibration after lighting changes.')
    return denominator


def normalize(dark,white,fruit):
    denominator=white['mean']-dark['mean']
    # Keep actual values, including >1. Do not silently clip or normalize to sum=1.
    with np.errstate(divide='ignore',invalid='ignore'):
        relative=(fruit['mean']-dark['mean'])/denominator
    return relative


def repeatability_table(scans):
    a=np.asarray(scans,dtype=float)
    mean=a.mean(axis=0);std=a.std(axis=0,ddof=1)
    cv=np.divide(100*std,np.abs(mean),out=np.full(8,np.nan),where=np.abs(mean)>1e-6)
    result=pd.DataFrame({'channel':CHANNELS,**{f'scan_{i+1}':a[i] for i in range(len(a))},
                         'mean':mean,'std':std,'range':np.ptp(a,axis=0),'cv_percent':cv})
    return result


def jsonable(value):
    if isinstance(value,np.ndarray): return [jsonable(x) for x in value.tolist()]
    if isinstance(value,dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [jsonable(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
    if isinstance(value,np.generic): return value.item()
    return value


class Log:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.raw=(self.directory/'serial.jsonl').open('a',buffering=1)
        self.events=(self.directory/'measurements.jsonl').open('a',buffering=1)
    def line(self,phase,line):
        self.raw.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'phase':phase,'line':line})+'\n')
    def event(self,**data):
        data['utc']=datetime.now(timezone.utc).isoformat()
        self.events.write(json.dumps(jsonable(data),allow_nan=False)+'\n')
    def close(self): self.raw.close();self.events.close()


class Reader:
    def __init__(self,port,log,timeout=45):
        self.port=port;self.log=log;self.timeout=timeout;self.settings=None;self.last_seq=None
    def collect(self,phase,n,discard=2):
        # Discard old buffered and in-flight readings after the user finishes positioning.
        self.port.reset_input_buffer();rows=[];deadline=time.monotonic()+self.timeout
        print(f'Collecting {phase}: waiting for {n} fresh readings...', flush=True)
        while len(rows)<n:
            if time.monotonic()>deadline:
                raise InvalidScan(f'Timed out after {len(rows)}/{n} readings. If zero: upload RipeSense_AS7341 to UNO R4 WiFi, close Serial Monitor, and check the current USB port.')
            line=self.port.readline().decode('utf-8',errors='replace').strip()
            if not line: continue
            self.log.line(phase,line)
            row=parse_line(line)
            if row is None: continue
            settings=tuple(row[k] for k in SETTINGS)
            if self.settings is None:self.settings=settings
            if settings!=self.settings: raise CalibrationChanged('Gain/timing/LED changed. Restart the whole dark/white/fruit session.')
            if self.last_seq is not None and row['seq']<=self.last_seq:
                raise CalibrationChanged('Board restarted or sequence repeated. Restart the whole calibration session.')
            self.last_seq=row['seq']
            if discard:discard-=1;continue
            rows.append(row)
            if len(rows)==1 or len(rows)%5==0 or len(rows)==n:
                print(f'  {phase}: {len(rows)}/{n} readings', flush=True)
        return rows


def measure(reader,log,phase,prompt,n,dark=None,white=None,min_counts=100,snr=10):
    attempt=0
    while True:
        input(prompt+' Press Enter when ready (Ctrl-C quits). ');attempt+=1
        summary=None;relative=None;rows=[]
        try:
            rows=reader.collect(phase,n);summary=summarize(rows)
            # Log normalized numbers even if this fruit scan subsequently fails QC.
            if dark is not None and white is not None:relative=normalize(dark,white,summary)
            check_raw(rows)
            if phase=='white':reference_quality(dark,summary,min_counts,snr)
            if relative is not None:
                if not np.isfinite(relative).all():raise InvalidScan('Nonfinite normalized scan; repeat calibration.')
                if np.any(relative < -0.02) or np.any(relative > 1.5):
                    raise InvalidScan('Relative reflectance outside bench guard [-0.02, 1.5]; check position, cover removal and reference, then repeat. Values were logged, not clipped.')
            log.event(phase=phase,attempt=attempt,valid=True,raw_summary=summary,relative_reflectance=relative,channel_order=CHANNELS)
            print(pd.DataFrame({'channel':CHANNELS,'mean_counts':summary['mean'],'std_counts':summary['std']}).round(3).to_string(index=False))
            return summary,relative
        except InvalidScan as e:
            log.event(phase=phase,attempt=attempt,valid=False,error=str(e),raw_summary=summary,relative_reflectance=relative)
            print('INVALID:',e,'\nRepeat this measurement; Ctrl-C to restart all calibration.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',help='e.g. /dev/cu.usbmodemF412FA6378E02 or COM4')
    parser.add_argument('--model',type=Path,default=Path('deephs_ripeness.joblib'))
    parser.add_argument('--fruit',default='apple',help='Selected model key; apple/unknown fruit is sensor-only')
    parser.add_argument('--samples',type=int,default=20)
    parser.add_argument('--scans',type=int,default=3,help='Repeat the SAME spot; >=2 for repeatability (default 3)')
    parser.add_argument('--min-reference-counts',type=float,default=100)
    parser.add_argument('--reference-snr',type=float,default=10)
    parser.add_argument('--log-dir',type=Path,default=Path('bench_logs'))
    parser.add_argument('--list-ports',action='store_true')
    parser.add_argument('--check-model',action='store_true',help='Validate bundle/schema without opening serial')
    args=parser.parse_args(argv)
    if args.list_ports:
        for p in list_ports.comports():print(f'{p.device}\t{p.description}\t{p.hwid}')
        return 0
    if args.samples<2 or args.scans<2 or args.min_reference_counts<=0 or args.reference_snr<=0:
        parser.error('samples/scans must be >=2; reference thresholds must be positive')
    bundle=load_bundle(args.model)
    print('Model fruits:',', '.join(bundle['models']));print('Saved feature order:',bundle['feature_order'])
    if args.check_model:return 0
    if not args.port:parser.error('--port is required (use --list-ports)')
    fruit=args.fruit.lower().strip()
    print('EXPERIMENTAL live-sensor result. Calibration approximates camera features; live accuracy is unvalidated.')
    if fruit not in bundle['models']:print(f'No {fruit} ripeness model in this bundle: raw and normalized sensor-only scans will still be logged.')
    print('Keep illumination, gain/timing, sensor angle and distance FIXED. This session does not change firmware settings.')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    log=Log(args.log_dir/stamp);print('Logs:',log.directory)
    log.event(phase='session',fruit=fruit,model=str(args.model.resolve()),model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
              feature_order=bundle['feature_order'],samples=args.samples,scans=args.scans,
              min_reference_counts=args.min_reference_counts,reference_snr=args.reference_snr)
    try:
        with serial.Serial(args.port,115200,timeout=1,exclusive=True) as port:
            time.sleep(2)  # Some boards reset when their USB serial port opens.
            reader=Reader(port,log,timeout=max(45,args.samples*3))
            dark,_=measure(reader,log,'dark','DARK: cover the sensor aperture completely; leave scan lighting on.',args.samples)
            white,_=measure(reader,log,'white','WHITE: uncover sensor; place a diffuse white reference at the fruit position under scan lighting.',
                            args.samples,dark=dark,min_counts=args.min_reference_counts,snr=args.reference_snr)
            log.event(phase='calibration',settings=dict(zip(SETTINGS,reader.settings)),dark=dark,white=white,
                      white_minus_dark=white['mean']-dark['mean'])
            values=[]
            for i in range(args.scans):
                _,relative=measure(reader,log,f'fruit_{i+1}',f'FRUIT {i+1}/{args.scans}: aim at the SAME spot of the SAME {fruit}, at the white-reference distance.',
                                   args.samples,dark=dark,white=white)
                values.append(relative);result=predict(bundle,fruit,relative)
                print('Relative reflectance:',dict(zip(CHANNELS,np.round(relative,5))))
                print(json.dumps(result,indent=2));log.event(phase='prediction',scan=i+1,relative_reflectance=relative,**result)
            table=repeatability_table(values)
            print('\nSame-spot repeatability (normalized channel units; sample SD):')
            print(table.round(5).to_string(index=False))
            table.to_csv(log.directory/'repeatability.csv',index=False)
            log.event(phase='repeatability',rows=table.to_dict('records'))
        return 0
    except (CalibrationChanged,serial.SerialException) as e:
        log.event(phase='aborted',error=str(e));print('STOP:',e);return 1
    except (KeyboardInterrupt,EOFError):
        log.event(phase='aborted',error='User ended session');print('\nStopped; partial readings retained.');return 130
    finally:log.close()


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileNotFoundError) as e:raise SystemExit(str(e))
