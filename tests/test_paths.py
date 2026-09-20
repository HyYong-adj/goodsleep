import hashlib
import importlib

import pytest

from psg_only import paths
from psg_only.paths import REPO_ROOT, resolve


def test_legacy_external_paths_rewrite_into_vendor():
    """옛 절대경로는 사본을 가리켜야 한다. 보관된 run config 19 개가 여기 걸려 있다."""
    vendor = paths.vendor_root()
    assert resolve("/home/sleep/researchers/byoungjun/cache/audio_compact_full") == vendor / "cache/audio_compact_full"
    assert resolve("/home/sleep/researchers/byoungjun/manifests/teacher_full.csv") == vendor / "manifests/teacher_full.csv"
    assert resolve("/home/sleep/data/splits/psg_audio_subject_split_v1.csv") == vendor / "splits/psg_audio_subject_split_v1.csv"


def test_own_absolute_path_follows_the_repository():
    """계정을 다른 위치로 복원해도 자기 산출물을 따라가야 한다."""
    assert resolve("/home/sleep/researchers/choihy/artifacts/x") == REPO_ROOT / "artifacts/x"


def test_raw_root_is_not_redirected_into_vendor():
    """원본 919GB 는 복사 대상이 아니다. 마운트 위치만 바뀐다."""
    assert resolve("/home/sleep/data/psg_audio/V3/a.edf") == paths.raw_root() / "V3/a.edf"


def test_placeholders_expand():
    assert resolve("${VENDOR}/cache/x") == paths.vendor_root() / "cache/x"
    assert resolve("${REPO}/artifacts/x") == REPO_ROOT / "artifacts/x"
    assert resolve("${RAW}/V3") == paths.raw_root() / "V3"


def test_relative_paths_resolve_against_repository_not_cwd(tmp_path, monkeypatch):
    """chain 스크립트가 cd 하지 않아도 같은 파일을 가리켜야 한다."""
    monkeypatch.chdir(tmp_path)
    assert resolve("artifacts/teacher_canonical") == REPO_ROOT / "artifacts/teacher_canonical"


def test_resolution_is_idempotent():
    once = resolve("/home/sleep/researchers/byoungjun/cache/audio_compact_full")
    assert resolve(once) == once


def test_environment_override_moves_the_vendor_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PSG_VENDOR_ROOT", str(tmp_path))
    assert resolve("/home/sleep/researchers/byoungjun/cache/x") == tmp_path / "cache/x"


def test_mel_cache_basename_is_pinned():
    """embedding provenance 가 이 basename 을 기록한다. 바꾸면 기존 캐시가 거부된다."""
    assert paths.MEL_CACHE_NAME == "audio_compact_full"
    assert paths.mel_cache().name == "audio_compact_full"


def test_verify_split_rejects_a_tampered_copy(tmp_path, monkeypatch):
    """split membership 이 조용히 바뀌면 모든 비교가 무효가 된다."""
    splits = tmp_path / "splits"
    splits.mkdir()
    csv = splits / "psg_audio_subject_split_v1.csv"
    csv.write_text("subject_id,split\na,train\n", encoding="utf-8")
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    (splits / "psg_audio_subject_split_v1.sha256").write_text(
        f"{digest}  psg_audio_subject_split_v1.csv\n", encoding="utf-8")

    monkeypatch.setenv("PSG_VENDOR_ROOT", str(tmp_path))
    assert paths.verify_split() is True

    csv.write_text("subject_id,split\na,test_locked\n", encoding="utf-8")
    with pytest.raises(ValueError):
        paths.verify_split()
    assert paths.verify_split(strict=False) is False


def test_verify_split_reports_a_missing_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("PSG_VENDOR_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        paths.verify_split()


def test_installed_copy_matches_the_approved_split():
    """실제 사본이 승인본과 같은지. 사본을 쓰는 이상 이 검사는 켜 둔다."""
    importlib.reload(paths)
    assert paths.verify_split() is True
