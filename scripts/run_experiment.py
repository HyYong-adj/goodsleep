"""Prepare or execute isolated B0 -> embedding -> B1 development experiments."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from psg_only.data import validate_inputs
from psg_only.raw import relocated_manifest, audit_raw
from psg_only.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b0-config', type=Path, default=ROOT/'configs/b0_epoch.yaml')
    parser.add_argument('--b1-config', type=Path, default=ROOT/'configs/b1_40to20.yaml')
    parser.add_argument('--parent-run', type=Path, help='Compare a B0-LR ablation against this completed full run.')
    parser.add_argument('--source', choices=['cached', 'raw'], default='cached')
    parser.add_argument('--raw-root', type=Path, default=Path('/home/sleep/data/psg_audio'))
    parser.add_argument('--manifest', type=Path, default=Path('/home/sleep/researchers/byoungjun/manifests/teacher_full.csv'))
    parser.add_argument('--cache-root', type=Path, default=Path('/home/sleep/researchers/byoungjun/cache/audio_compact_full'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run', action='store_true', help='Execute after preparing; otherwise only validate and write resolved configs.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', help='GPU index explicitly assigned to this run')
    group.add_argument('--cpu', action='store_true')
    parser.add_argument('--raw-python', type=Path, default=ROOT / '.venv-raw/bin/python')
    args = parser.parse_args()
    if args.run and args.gpu is None and not args.cpu:
        parser.error('--run requires --gpu ASSIGNED_INDEX or --cpu')
    if args.parent_run:
        import torch
        parent_cfg = torch.load(args.parent_run/'b0/best.pt',map_location='cpu',weights_only=False)['config']
        if args.source != 'cached' or parent_cfg.get('smoke') or args.seed != parent_cfg['seed']:
            parser.error('LR ablation requires cached source, a full parent, and the same seed.')
        candidate = yaml.safe_load(args.b0_config.read_text())
        expected_train = dict(parent_cfg['train'], learning_rate=candidate['train']['learning_rate'])
        parent_b1 = yaml.safe_load((args.parent_run/'b1.yaml').read_text())
        candidate_b1 = yaml.safe_load(args.b1_config.read_text())
        if candidate['model'] != parent_cfg['model'] or candidate['train'] != expected_train:
            parser.error('Only B0 learning_rate may change.')
        if any(candidate_b1[k] != parent_b1[k] for k in ('model','train')) or any(candidate_b1['data'].get(k) != parent_b1['data'].get(k) for k in ('input_epochs','target_epochs','left_context','stride')):
            parser.error('Keep the parent B1 model, training and context unchanged.')
        args.manifest = Path(parent_cfg['data']['manifest'])
        args.cache_root = Path(parent_cfg['data']['cache_root'])
        if file_hash(args.manifest) != parent_cfg['manifest_hash'] or file_hash(args.cache_root/'train_stats.json') != parent_cfg['stats_hash']:
            parser.error('Parent manifest or normalization changed.')
    out = (args.output or ROOT / 'artifacts/experiments' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='PREPARING', source=args.source, seed=args.seed, smoke=args.smoke,
                 evaluation_scope='validation-only', started_at=datetime.now(timezone.utc).isoformat(),
                 hypothesis='Frozen B0 acoustic embeddings plus 40-to-20 temporal context improve validation Macro-F1.',
                 decision_metrics=['macro_f1','cohen_kappa','per_class','transition_macro_f1','transition_rate_error'],
                 budget={'b0_epochs':1 if args.smoke else 30,'b1_epochs':1 if args.smoke else 50},
                 source_manifest_hash=file_hash(args.manifest))
    if args.parent_run:
        state.update(parent_run=str(args.parent_run.resolve()),
                     hypothesis='Reducing B0 LR from 3e-4 to 1e-4 improves B0 and downstream B1 40-to-20 validation Macro-F1.',
                     changed_parameter='b0.train.learning_rate',
                     parent_learning_rate=parent_cfg['train']['learning_rate'],
                     learning_rate=candidate['train']['learning_rate'])
    def save():
        (out / 'experiment.json').write_text(json.dumps(state, indent=2))
    save()
    try:
        frame = relocated_manifest(args.manifest, args.raw_root)
        if args.smoke:
            ids = set()
            for split, count in [('train',8),('val',2)]:
                ids.update(sorted(frame.loc[frame.split == split, 'subject_id'].unique())[:count])
            frame = frame.loc[frame.subject_id.isin(ids)].copy()
        manifest = out / 'manifest.csv'
        frame.to_csv(manifest, index=False)
        report = audit_raw(frame, args.raw_root)
        (out / 'raw_readiness.json').write_text(json.dumps(report, indent=2))
        cache = args.cache_root.resolve() if args.source == 'cached' else out / 'mel_cache'
        configs = {}
        for name, template in [('b0',args.b0_config),('b1',args.b1_config)]:
            cfg = yaml.safe_load(template.read_text())
            state['budget'][name+'_epochs'] = 1 if args.smoke else cfg['train']['epochs']
            cfg['seed'] = args.seed
            cfg['data']['manifest'] = str(manifest)
            if name == 'b0':
                cfg['data'].update(cache_root=str(cache), stats=str(cache / 'train_stats.json'))
            else:
                cfg['data']['embedding_root'] = str(out / 'embeddings')
            configs[name] = out / f'{name}.yaml'
            configs[name].write_text(yaml.safe_dump(cfg, sort_keys=False))
        state['raw_readiness'] = report['status']
        if args.source == 'cached':
            report_cache = validate_inputs(manifest, cache, cache / 'train_stats.json')
            (out / 'cache_preflight.json').write_text(json.dumps(report_cache, indent=2))
        elif report['status'] != 'ready':
            state['status'] = 'BLOCKED_RAW_COPY_OR_ALIGNMENT'
            save()
            print(json.dumps({'output':str(out), **state}, indent=2))
            return 2
        state['status'] = 'PREPARED'
        save()
        print(json.dumps({'output':str(out), 'status':state['status'], 'raw_status':report['status']}), flush=True)
        if not args.run:
            return 0
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if args.cpu else args.gpu,
                   OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS','4'), CUBLAS_WORKSPACE_CONFIG=':4096:8')
        def run(stage, argv, interpreter=sys.executable):
            state.update(status='RUNNING', stage=stage)
            save()
            with (out / f'{stage}.log').open('w') as log:
                process = subprocess.Popen([str(interpreter), *map(str,argv)], cwd=ROOT, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
                code = process.wait()
            if code:
                raise RuntimeError(f'{stage} exited with code {code}; see {stage}.log')
        if args.source == 'raw':
            run('raw_cache', ['scripts/cache_mels_raw.py','--manifest',manifest,'--output',cache,'--raw-root',args.raw_root], args.raw_python)
            (out / 'cache_preflight.json').write_text(json.dumps(validate_inputs(manifest,cache,cache/'train_stats.json'),indent=2))
        smoke = ['--smoke'] if args.smoke else []
        run('train_b0', ['scripts/train_b0.py','--config',configs['b0'],'--output',out/'b0',*smoke])
        run('evaluate_b0', ['scripts/evaluate.py','--checkpoint',out/'b0/best.pt','--output',out/'b0/evaluation'])
        run('embeddings', ['scripts/cache_embeddings.py','--checkpoint',out/'b0/best.pt','--output',out/'embeddings'])
        run('train_b1', ['scripts/train_b1.py','--config',configs['b1'],'--output',out/'b1',*smoke])
        run('evaluate_b1', ['scripts/evaluate.py','--checkpoint',out/'b1/best.pt','--output',out/'b1/evaluation'])
        import pandas as pd
        predictions = [pd.read_csv(out/name/'evaluation/predictions.csv',dtype={'subject_id':str})
                       .sort_values(['subject_id','epoch_index']).reset_index(drop=True) for name in ('b0','b1')]
        if not predictions[0][['subject_id','epoch_index','target']].equals(predictions[1][['subject_id','epoch_index','target']]):
            raise ValueError('B0/B1 validation target sets differ.')
        results = {name:json.loads((out/name/'evaluation/results.json').read_text()) for name in ('b0','b1')}
        if any(r['status'] != 'complete' for r in results.values()):
            raise ValueError('Invalid evaluation: missing class support or predictions.')
        comparison = dict(evaluation_scope='validation-only', smoke=args.smoke, seed=args.seed,
                          identical_validation_epochs=True, epochs=len(predictions[0]),
                          metrics={name:r['metrics'] for name,r in results.items()},
                          macro_f1_delta=results['b1']['metrics']['macro_f1']-results['b0']['metrics']['macro_f1'])
        (out / 'comparison.json').write_text(json.dumps(comparison,indent=2))
        if args.parent_run and not args.smoke:
            parent_results = {}
            for name in ('b0','b1'):
                old = pd.read_csv(args.parent_run/name/'evaluation/predictions.csv',dtype={'subject_id':str})
                new = pd.read_csv(out/name/'evaluation/predictions.csv',dtype={'subject_id':str})
                keys=['subject_id','epoch_index','target']
                if not old[keys].sort_values(keys).reset_index(drop=True).equals(new[keys].sort_values(keys).reset_index(drop=True)):
                    raise ValueError('Parent/candidate validation targets differ.')
                parent_results[name] = json.loads((args.parent_run/name/'evaluation/results.json').read_text())
            payload = dict(parent_run=str(args.parent_run.resolve()), evaluation_scope='validation-only',
                           identical_validation_epochs=True, seed=args.seed,
                           metrics={name:{'parent':parent_results[name]['metrics'],'candidate':results[name]['metrics'],
                                          'macro_f1_delta':results[name]['metrics']['macro_f1']-parent_results[name]['metrics']['macro_f1']} for name in ('b0','b1')})
            (out/'parent_comparison.json').write_text(json.dumps(payload,indent=2))
        state['status'] = 'SMOKE_COMPLETE' if args.smoke else 'COMPLETE' 
        save()
        print(json.dumps({'output':str(out),'status':state['status']}))
        return 0
    except BaseException as exc:
        state.update(status='FAILED',error_type=type(exc).__name__)
        save()
        raise

if __name__ == '__main__':
    raise SystemExit(main())
