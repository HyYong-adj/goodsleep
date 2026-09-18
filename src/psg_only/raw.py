"""V3 intake audit; independent adaptation of sleepteacher/data/psg_audio.py.

Uses existing split and epoch annotations, never discovers new training members.
Only UserStaging labels are checked. No raw paths or participant identifiers are
included in the public summary. EDF checks are structural, not transfer checksums.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .data import read_manifest


def relocated_manifest(manifest, raw_root):
    frame = read_manifest(manifest)
    required = {'edf_path','channel_index','sample_rate','local_start_seconds','recording_start_seconds'}
    if not required.issubset(frame.columns):
        raise ValueError('Raw processing requires source EDF/timing columns.')
    root = Path(raw_root) / 'V3' / 'APNEA_EDF'
    frame['edf_path'] = [str(root / sid / Path(old).name) for sid, old in zip(frame.subject_id, frame.edf_path)]
    return frame


def audit_raw(frame, raw_root):
    import pyedflib
    from datetime import timedelta
    counts = dict(expected_edf=int(frame.edf_path.nunique()), checked_edf=0, missing_edf=0,
                  invalid_edf=0, expected_rml=int(frame.subject_id.nunique()), missing_rml=0,
                  invalid_rml=0, checked_rml=0, annotation_mismatches=0)
    stage_map = {'Wake':0, 'NonREM1':1, 'NonREM2':1, 'NonREM3':2, 'REM':3, 'NotScored':-100}
    ns = '{http://www.respironics.com/PatientStudy.xsd}'
    for sid, rows in frame.groupby('subject_id'):
        rml = Path(raw_root) / 'V3' / 'APNEA_RML' / f'{sid}.rml'
        if not rml.exists():
            counts['missing_rml'] += 1
        else:
            try:
                before = rml.stat()
                root = ET.parse(rml).getroot()
                staging = root.find(f'.//{ns}ScoringData/{ns}StagingData/{ns}UserStaging/{ns}NeuroAdultAASMStaging')
                if staging is None:
                    raise ValueError('Missing UserStaging.')
                transitions = sorted((float(s.attrib['Start']), stage_map[s.attrib['Type']]) for s in staging.findall(f'{ns}Stage'))
                starts = np.asarray([s[0] for s in transitions])
                if not len(starts) or np.any(np.diff(starts) <= 0):
                    raise ValueError('Invalid staging transitions.')
                times = rows.recording_start_seconds.to_numpy()
                ix = np.searchsorted(starts, times, side='right') - 1
                end_ix = np.searchsorted(starts, times + 30 - 1e-7, side='right') - 1
                if (ix < 0).any() or not np.array_equal(ix, end_ix):
                    raise ValueError('Epoch crosses an annotation transition.')
                labels = np.asarray([s[1] for s in transitions])[ix]
                counts['annotation_mismatches'] += int(np.count_nonzero(labels != rows.label_index.to_numpy()))
                after = rml.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('RML changed while checking.')
                counts['checked_rml'] += 1
            except (ValueError, KeyError, ET.ParseError, OSError):
                counts['invalid_rml'] += 1
        origin = None
        for name, epochs in rows.groupby('edf_path', sort=True):
            path = Path(name)
            if not path.exists():
                counts['missing_edf'] += 1
                continue
            try:
                before = path.stat()
                # EDF advertised byte length detects truncated transfers even if a
                # reader tolerates incomplete data records.
                with path.open('rb') as stream:
                    header = stream.read(256)
                    header_bytes = int(header[184:192]); records = int(header[236:244]); signals = int(header[252:256])
                    signal_header = stream.read(256 * signals)
                samples = [int(signal_header[216*signals+8*i:216*signals+8*(i+1)]) for i in range(signals)]
                if records < 0 or before.st_size != header_bytes + 2 * records * sum(samples):
                    raise ValueError('Incomplete EDF transfer or unsupported EDF size.')
                with pyedflib.EdfReader(str(path)) as reader:
                    channels = reader.getSignalLabels()
                    for row in epochs.itertuples(index=False):
                        ci = int(row.channel_index)
                        if channels[ci].strip() not in ('Mic', 'Microphone'):
                            raise ValueError('Expected ambient Mic channel.')
                        rate = reader.getSampleFrequency(ci)
                        if rate != float(row.sample_rate) or row.local_start_seconds < 0 or row.local_start_seconds + 30 > reader.getFileDuration() + 1e-6:
                            raise ValueError('Sample rate or epoch bounds mismatch.')
                        this_origin = reader.getStartdatetime() - timedelta(seconds=float(row.recording_start_seconds - row.local_start_seconds))
                        if origin is None:
                            origin = this_origin
                        if this_origin != origin:
                            raise ValueError('EDF timestamp/manifest alignment mismatch.')
                    # Read a short sample at start/middle/end, avoiding a full raw scan.
                    ci = int(epochs.channel_index.iloc[0]); n = int(reader.getNSamples()[ci])
                    for start in (0, n//2, max(0, n-1024)):
                        signal = reader.readSignal(ci, start=start, n=min(1024,n-start))
                        if not len(signal) or not np.isfinite(signal).all():
                            raise ValueError('Invalid Mic samples.')
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('EDF changed while checking.')
                counts['checked_edf'] += 1
            except (OSError, ValueError, IndexError):
                counts['invalid_edf'] += 1
    counts['status'] = 'ready' if not any(counts[k] for k in ('missing_edf','invalid_edf','missing_rml','invalid_rml','annotation_mismatches')) else 'not_ready'
    counts['verification'] = 'EDF length/header/timestamps and sampled Mic reads; RML UserStaging vs existing manifest; no source transfer checksum'
    return counts
