import json
import numpy as np
import pandas as pd
import pytest
import torch
from test_data import make_fixture
from psg_only.data import validate_inputs, read_manifest, CachedMelEpochDataset, EmbeddingWindowDataset
from psg_only.constants import CANONICAL_CLASSES, CLASS_ORDER_VERSION
from psg_only.provenance import file_hash
from psg_only.windows import stitch_predictions
from psg_only.train import load_config, _checkpoint, seed_everything
from psg_only.evaluate import load_model
from psg_only.models import B1Model


def embedding_fixture(tmp_path):
    manifest, cache, stats = make_fixture(tmp_path)
    frame = pd.read_csv(manifest)
    frame['recording_start_seconds'] = frame.subject_epoch_index * 30
    frame.loc[frame.subject_epoch_index >= 2, 'recording_start_seconds'] += 30
    frame.to_csv(manifest,index=False)
    root = tmp_path/'embeddings'; directory = root/'s1'; directory.mkdir(parents=True)
    np.save(directory/'embeddings.npy',np.zeros((5,192),np.float16))
    np.save(directory/'labels.npy',np.array([0,2,-100,3,1],np.int64))
    np.save(directory/'valid.npy',np.array([True,True,False,True,True]))
    metadata=dict(checkpoint_sha256='fixture',config_hash='fixture',stats_hash='fixture',source_cache='fixture',
                  class_order=list(CANONICAL_CLASSES),class_order_version=CLASS_ORDER_VERSION,
                  manifest_hash=file_hash(manifest),subject_id='s1',
                  files={n:file_hash(directory/n) for n in ('embeddings.npy','labels.npy','valid.npy')})
    (directory/'metadata.json').write_text(json.dumps(metadata))
    return manifest,root,directory,metadata


def test_manifest_cache_mismatch_is_rejected(tmp_path):
    manifest,root,stats=make_fixture(tmp_path)
    np.save(root/'s1/labels.npy',np.array([3,1,2,3]))
    with pytest.raises(ValueError,match='label mismatch'):
        validate_inputs(manifest,root,stats)


def test_training_dataset_rejects_leakage(tmp_path):
    manifest,root,stats=make_fixture(tmp_path)
    frame=pd.read_csv(manifest);frame.loc[4,'subject_id']='s1';frame.to_csv(manifest,index=False)
    with pytest.raises(ValueError,match='leakage'):
        CachedMelEpochDataset(manifest,root,stats,'train')


def test_real_timeline_preserves_unscored_gap(tmp_path):
    manifest,root,directory,metadata=embedding_fixture(tmp_path)
    ds=EmbeddingWindowDataset(manifest,root,'train')
    assert ds.expected_keys()=={('s1',0),('s1',1),('s1',3),('s1',4)}
    x,y,iv,tv,sid,idx=ds[0]
    assert idx[tv].tolist()==[0,1,3,4]
    assert not iv[12]
    f=read_manifest(manifest)
    assert f.loc[f.subject_id=='s1','epoch_index'].tolist()==[0,1,3,4]


def test_missing_or_corrupt_embedding_provenance_fails(tmp_path):
    manifest,root,directory,metadata=embedding_fixture(tmp_path)
    (directory/'metadata.json').unlink()
    with pytest.raises(FileNotFoundError):
        EmbeddingWindowDataset(manifest,root,'train')
    (directory/'metadata.json').write_text(json.dumps(metadata))
    np.save(directory/'embeddings.npy',np.ones((5,192),np.float16))
    with pytest.raises(ValueError,match='hash mismatch'):
        EmbeddingWindowDataset(manifest,root,'train')


def test_extra_embedding_epoch_rejected(tmp_path):
    manifest,root,directory,metadata=embedding_fixture(tmp_path)
    np.save(directory/'valid.npy',np.ones(5,bool))
    metadata['files']['valid.npy']=file_hash(directory/'valid.npy')
    (directory/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='outside manifest'):
        EmbeddingWindowDataset(manifest,root,'train')


def test_stitch_rejects_missing_prediction():
    with pytest.raises(ValueError,match='coverage mismatch'):
        stitch_predictions([{'subject_id':'s','epoch_index':0}],{('s',0),('s',1)})


def test_checkpoint_roundtrip_and_seed(tmp_path):
    seed_everything(4); model=B1Model().eval()
    x=torch.randn(1,40,192)
    expected=model(x)
    cfg={'model':{'hidden_dim':128,'layers':2,'dropout':0.2}}
    path=tmp_path/'best.pt';_checkpoint(path,model,cfg,'b1',0,0.0)
    restored,payload=load_model(path,torch.device('cpu'))
    assert torch.equal(expected,restored(x))
    seed_everything(4); second=B1Model().eval()
    assert torch.equal(expected,second(x))
    payload.pop('class_order');torch.save(payload,path)
    with pytest.raises(ValueError,match='class order'):
        load_model(path,torch.device('cpu'))


def test_fixed_config_rejects_ignored_context(tmp_path):
    import yaml
    config=yaml.safe_load(open('configs/b1_40to20.yaml'))
    config['data']['stride']=40
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='stride'):
        load_config(path)


def test_raw_audit_detects_missing_copy(tmp_path):
    from psg_only.raw import relocated_manifest,audit_raw
    manifest,root,stats=make_fixture(tmp_path)
    f=pd.read_csv(manifest)
    f['edf_path']='/old/missing.edf';f['channel_index']=0;f['sample_rate']=48000
    f['local_start_seconds']=f.subject_epoch_index*30;f['recording_start_seconds']=f.local_start_seconds
    f.to_csv(manifest,index=False)
    relocated=relocated_manifest(manifest,tmp_path)
    report=audit_raw(relocated,tmp_path)
    assert report['status']=='not_ready' and report['missing_edf']==2 and report['missing_rml']==2


def test_raw_audit_checks_edf_and_user_staging(tmp_path):
    import pyedflib
    from psg_only.raw import audit_raw
    directory=tmp_path/'V3/APNEA_EDF/synthetic';directory.mkdir(parents=True)
    path=directory/'synthetic[001].edf'
    header=pyedflib.highlevel.make_signal_header('Mic',sample_frequency=48)
    pyedflib.highlevel.write_edf(str(path),np.zeros((1,48*120)),[header])
    rml_dir=tmp_path/'V3/APNEA_RML';rml_dir.mkdir()
    rml=rml_dir/'synthetic.rml'
    stages=''.join(f'<Stage Start="{i*30}" Type="{name}"/>' for i,name in enumerate(['Wake','NonREM2','NonREM3','REM']))
    rml.write_text('<Root xmlns="http://www.respironics.com/PatientStudy.xsd"><ScoringData><StagingData><UserStaging><NeuroAdultAASMStaging>'+stages+'</NeuroAdultAASMStaging></UserStaging></StagingData></ScoringData></Root>')
    frame=pd.DataFrame([dict(subject_id='synthetic',edf_path=str(path),channel_index=0,sample_rate=48,
                             local_start_seconds=i*30,recording_start_seconds=i*30,label_index=i) for i in range(4)])
    assert audit_raw(frame,tmp_path)['status']=='ready'
    frame.loc[0,'label_index']=3
    assert audit_raw(frame,tmp_path)['annotation_mismatches']==1
    with path.open('r+b') as stream:
        stream.truncate(path.stat().st_size-10)
    assert audit_raw(frame,tmp_path)['invalid_edf']==1


def test_b1_skips_all_invalid_batch_and_repeats_seed(tmp_path,monkeypatch):
    from psg_only.train import train_b1
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    rows=[]; root=tmp_path/'embeddings'
    for split,sid in [('train','a'),('val','b')]:
        directory=root/sid;directory.mkdir(parents=True)
        np.save(directory/'embeddings.npy',np.zeros((40,192),np.float16))
        valid=np.arange(40)>=20
        labels=np.asarray((0,2,3,1))[np.arange(40)%4];labels[~valid]=-100
        np.save(directory/'labels.npy',labels);np.save(directory/'valid.npy',valid)
        rows.extend(dict(subject_id=sid,split=split,subject_epoch_index=i,label_index=i%4) for i in range(40))
    manifest=tmp_path/'manifest.csv';pd.DataFrame(rows).to_csv(manifest,index=False)
    for directory in root.iterdir():
        (directory/'metadata.json').write_text(json.dumps(dict(checkpoint_sha256='fixture',config_hash='fixture',stats_hash='fixture',source_cache='fixture',class_order=list(CANONICAL_CLASSES),class_order_version=CLASS_ORDER_VERSION,manifest_hash=file_hash(manifest),subject_id=directory.name,files={n:file_hash(directory/n) for n in ('embeddings.npy','labels.npy','valid.npy')})))
    config={'seed':5,'data':{'manifest':str(manifest),'embedding_root':str(root)},'model':{'hidden_dim':4,'layers':1,'dropout':0.2},'train':{'batch_size':1,'learning_rate':0.001,'weight_decay':0.,'grad_clip_norm':1.,'epochs':1,'early_stopping_patience':1}}
    a=train_b1(config,tmp_path/'run_a',smoke=True)
    b=train_b1(config,tmp_path/'run_b',smoke=True)
    state_a=torch.load(a,weights_only=False)['state_dict'];state_b=torch.load(b,weights_only=False)['state_dict']
    assert all(torch.equal(state_a[k],state_b[k]) for k in state_a)
    assert json.loads((tmp_path/'run_a/history.jsonl').read_text())['optimizer_steps']==1
