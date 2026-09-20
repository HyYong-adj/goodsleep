# Frozen subject splits

| File | Subjects | Split counts | sha256 |
|---|---|---|---|
| psg_audio_subject_split_v1.csv | 284 | {'train': 191, 'test_locked': 49, 'val': 44} | fb6dff988c638ee1… |

- Source: PSG-Audio (Korompili et al. 2021), 287 patients; 284 passed strict checks (3 excluded).
- Origin of membership: split used for the PSG teacher and all student experiments since 2026-09-09 (`researchers/byoungjun/manifests/teacher_full.csv` train/val; `test_locked` from `reports/label_audit/hypnogram_label_coverage_20260910.csv`).
- Rule: `test_locked` subjects must not be used for training, model/hyperparameter selection, or interim checks. Every evaluation on them is recorded in `registry/TEST_ACCESS.md`.
- Regenerating any manifest must reproduce this membership exactly; verify with `sha256sum -c psg_audio_subject_split_v1.sha256`.
