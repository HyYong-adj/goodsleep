"""Train/evaluate a context ablation on the parent run's frozen B0 embeddings."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import math
import torch
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from psg_only.data import EmbeddingWindowDataset
from psg_only.train import load_config, train_b1, device_for_run
from psg_only.evaluate import evaluate_b1, write_evaluation
from psg_only.provenance import file_hash
from psg_only.windows import context_options


def preflight(cfg, smoke=False):
    parent = Path(cfg['experiment']['parent_run'])
    old = torch.load(parent/'b1/best.pt', map_location='cpu', weights_only=False)['config']
    if old.get('smoke'):
        raise ValueError('Parent must be a full run.')
    if cfg['model'] != old['model'] or cfg['train'] != old['train'] or cfg['seed'] != old['seed']:
        raise ValueError('Only context may change; retain parent model/train/seed settings.')
    data = cfg['data']
    if file_hash(data['manifest']) != old['manifest_hash']:
        raise ValueError('Manifest differs from parent.')
    if Path(data['embedding_root']).resolve() != Path(old['data']['embedding_root']).resolve():
        raise ValueError('Reuse the parent embedding directory.')
    if file_hash(parent/'b0/best.pt') != old['embedding_provenance']['checkpoint_sha256']:
        raise ValueError('Parent B0 checkpoint differs from pinned embedding provenance.')
    report = dict(parent_run=str(parent), seed=cfg['seed'], parent_input_epochs=old['data']['input_epochs'],
                  input_epochs=data['input_epochs'], target_epochs=20,
                  left_context=(data['input_epochs']-20)//2, stride=20,
                  embedding_provenance=old['embedding_provenance'], splits={})
    for split, limit in [('train',8),('val',2)]:
        ds = EmbeddingWindowDataset(data['manifest'],data['embedding_root'],split,
                                    subject_limit=limit if smoke else None, **context_options(data))
        if ds.provenance != old['embedding_provenance']:
            raise ValueError('Embedding provenance differs from parent.')
        report['splits'][split] = dict(subjects=len(ds.groups), windows=len(ds), valid_epochs=len(ds.expected_keys()))
    report['max_optimizer_steps'] = math.ceil(report['splits']['train']['windows']/cfg['train']['batch_size'])*cfg['train']['epochs']
    report['budget_note'] = 'Same maximum epochs/updates and patience as parent; early stopping can produce different actual update counts.'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/b1_80to20.yaml')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    cfg=load_config(args.config)
    out=(args.output or ROOT/'artifacts/experiments'/datetime.now(timezone.utc).strftime('b1_context_%Y%m%dT%H%M%S%fZ')).resolve()
    out.mkdir(parents=True,exist_ok=False)
    state=dict(status='PREPARING',smoke=args.smoke,seed=cfg['seed'],evaluation_scope='validation-only',
               started_at=datetime.now(timezone.utc).isoformat(),**cfg['experiment'])
    def save():
        (out/'experiment.json').write_text(json.dumps(state,indent=2))
    save()
    (out/'b1.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    try:
        report=preflight(cfg,args.smoke)
        (out/'preflight.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(output=str(out),**report),indent=2),flush=True)
        if args.preflight_only:
            state['status']='PREPARED';save();return
        device=device_for_run()
        if not args.smoke and device.type != 'cuda':
            raise RuntimeError('Full context run requires a visible CUDA GPU; set CUDA_VISIBLE_DEVICES=3.')
        state.update(status='RUNNING',stage='train_b1',device=str(device));save()
        path=train_b1(cfg,out/'b1',smoke=args.smoke)
        state['stage']='evaluate_b1';save()
        rows,checkpoint=evaluate_b1(path)
        result=write_evaluation(rows,checkpoint,out/'b1/evaluation')
        if result['status'] != 'complete':
            raise ValueError('Evaluation does not have all four classes.')
        if not args.smoke:
            parent=Path(cfg['experiment']['parent_run'])
            old=pd.read_csv(parent/'b1/evaluation/predictions.csv',dtype={'subject_id':str})
            new=pd.DataFrame(rows)
            keys=['subject_id','epoch_index','target']
            old=old[keys].sort_values(keys).reset_index(drop=True)
            new=new[keys].sort_values(keys).reset_index(drop=True)
            if not old.equals(new):
                raise ValueError('Parent and candidate validation epochs/targets differ.')
            baseline=json.loads((parent/'b1/evaluation/results.json').read_text())
            comparison=dict(evaluation_scope='validation-only',seed=cfg['seed'],identical_validation_epochs=True,
                            epochs=len(new),parent_run=str(parent),
                            parent_input_epochs=report['parent_input_epochs'],input_epochs=report['input_epochs'],
                            metrics={'parent_b1':baseline['metrics'],'candidate_b1':result['metrics']},
                            macro_f1_delta=result['metrics']['macro_f1']-baseline['metrics']['macro_f1'])
            (out/'comparison.json').write_text(json.dumps(comparison,indent=2))
        state['status']='SMOKE_COMPLETE' if args.smoke else 'COMPLETE';save()
        print(json.dumps(dict(output=str(out),status=state['status'],macro_f1=result['metrics']['macro_f1'])),flush=True)
    except BaseException as error:
        state.update(status='FAILED',error_type=type(error).__name__);save();raise

if __name__=='__main__':
    main()
