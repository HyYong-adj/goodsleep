"""트랙이 한 계정 안에서 자족적으로 돌도록 경로를 해석한다.

이 트랙은 원래 byoungjun 소유의 mel 캐시·manifest·teacher logits 와 공용
`data/splits` 를 절대경로로 직접 읽었다. 계정을 통째로 옮기면 그 경로가 전부
끊긴다. 사본을 `vendor/` 에 두고, **파일을 실제로 여는 지점에서만** 경로를
바꿔치기한다.

왜 config 딕셔너리를 고치지 않는가
----------------------------------
`train.config_hash` 는 config 를 통째로 직렬화해 해시한다. 로드 시점에 config
안의 경로 문자열을 다시 써 버리면 해시가 달라지고, 기존 체크포인트의
자기 검증(`config_hash(cp["config"]) != cp["config_hash"]`)이 전부 깨진다.
그래서 config 는 건드리지 않고 `resolve()` 를 파일시스템 진입점에 넣는다.
덕분에 보관된 실험 디렉터리의 `c1.yaml`/`b1.yaml` 19 개도 **기록 그대로 둔 채**
동작한다. 그 파일들은 "무엇을 썼는가" 의 증거이므로 고쳐 쓰면 안 된다.

무결성 게이트에 영향이 없는 이유
--------------------------------
- `manifest_hash`/`stats_hash` 는 **내용** 해시다. 사본은 바이트 동일이라 일치한다.
- embedding provenance 의 `source_cache` 는 `Path(cache_root).name` 즉 basename
  뿐이다. 사본의 디렉터리 이름을 `audio_compact_full` 로 유지하는 한 동일하다.
  **이 이름을 바꾸면 기존 embedding 캐시와 provenance 가 어긋난다.**

새 config 는 `${VENDOR}` 를 쓴다. 그래야 config_hash 가 기계마다 달라지지 않는다.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: 사본 디렉터리의 basename. embedding provenance 가 이 값에 묶여 있다.
MEL_CACHE_NAME = "audio_compact_full"


def vendor_root() -> Path:
    """복사해 온 외부 자산의 뿌리. `PSG_VENDOR_ROOT` 로 덮어쓸 수 있다."""
    override = os.environ.get("PSG_VENDOR_ROOT")
    return Path(override) if override else REPO_ROOT / "vendor"


def raw_root() -> Path:
    """원본 PSG-Audio(919GB). 복사 대상이 아니라 마운트 위치만 바꾼다."""
    return Path(os.environ.get("PSG_RAW_ROOT", "/home/sleep/data/psg_audio"))


def mel_cache() -> Path:
    return vendor_root() / "cache" / MEL_CACHE_NAME


def train_stats_path() -> Path:
    return mel_cache() / "train_stats.json"


def teacher_targets() -> Path:
    """PSG teacher 원본 logits. `build_teacher_cache.py` 의 입력이다."""
    return vendor_root() / "cache" / "teacher_targets"


def manifests() -> Path:
    return vendor_root() / "manifests"


def splits() -> Path:
    return vendor_root() / "splits"


def split_csv() -> Path:
    return splits() / "psg_audio_subject_split_v1.csv"


# 긴 접두사를 먼저 봐야 `/home/sleep/data/psg_audio` 와 `/home/sleep/data/splits`
# 처럼 뿌리를 공유하는 항목이 서로를 가리지 않는다.
_LEGACY: tuple[tuple[str, "callable[[], Path]"], ...] = (
    ("/home/sleep/researchers/byoungjun/cache", lambda: vendor_root() / "cache"),
    ("/home/sleep/researchers/byoungjun/manifests", manifests),
    ("/home/sleep/data/psg_audio", raw_root),
    ("/home/sleep/data/splits", splits),
    # 이 트랙 자신의 절대경로. 계정을 다른 위치로 복원해도 따라온다.
    ("/home/sleep/researchers/choihy", lambda: REPO_ROOT),
)

_PLACEHOLDERS: dict[str, "callable[[], Path]"] = {
    "${VENDOR}": vendor_root,
    "${REPO}": lambda: REPO_ROOT,
    "${RAW}": raw_root,
}


def resolve(path: str | Path | None) -> Path | None:
    """설정에 적힌 경로를 이 계정 안의 실제 경로로 바꾼다.

    멱등이다. 이미 해석된 경로를 다시 넣어도 같은 값이 나온다.
    상대경로는 CWD 가 아니라 **저장소 뿌리** 기준으로 푼다. chain 스크립트가
    `cd` 를 하지 않아도 같은 파일을 가리키게 하려는 것이다.
    """
    if path is None:
        return None
    text = str(path)

    for token, base in _PLACEHOLDERS.items():
        if token in text:
            text = text.replace(token, str(base()))

    for prefix, base in sorted(_LEGACY, key=lambda item: -len(item[0])):
        if text == prefix or text.startswith(prefix + "/"):
            text = str(base()) + text[len(prefix):]
            break

    resolved = Path(text)
    return resolved if resolved.is_absolute() else (REPO_ROOT / resolved)


def verify_split(strict: bool = True) -> bool:
    """동결 split 사본이 승인본과 같은지 확인한다.

    split membership 이 조용히 바뀌면 모든 비교가 무효가 되고, `test_locked` 가
    오염되면 되돌릴 수 없다. 사본을 쓰는 이상 이 검사는 켜 둔다.
    """
    import hashlib

    csv_path, digest_path = split_csv(), splits() / "psg_audio_subject_split_v1.sha256"
    if not csv_path.is_file() or not digest_path.is_file():
        if strict:
            raise FileNotFoundError(f"동결 split 사본이 없다: {csv_path}")
        return False
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if actual != expected and strict:
        raise ValueError(f"split 해시 불일치: {csv_path} (기대 {expected[:12]}…, 실제 {actual[:12]}…)")
    return actual == expected
