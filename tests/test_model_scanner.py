import os

import pytest

from app.core.model_scanner import scan


def _touch(p, size=16):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * size)


def test_scan_basic_and_mmproj(tmp_path):
    snap = tmp_path / "models--x" / "snapshots" / "abc"
    _touch(snap / "model-Q8_0.gguf")
    _touch(snap / "mmproj-BF16.gguf")
    res = scan([str(tmp_path)], with_metadata=False)
    assert len(res.models) == 1                     # mmproj not listed as a model
    m = res.models[0]
    assert m.name == "model-Q8_0.gguf"
    assert len(m.mmproj_candidates) == 1


def test_dedupe_by_blob(tmp_path):
    blobs = tmp_path / "blobs"
    blob = blobs / "sha256-deadbeef"
    _touch(blob, size=128)
    snap = tmp_path / "snapshots" / "v1"
    snap.mkdir(parents=True)
    link1 = snap / "model.gguf"
    link2 = tmp_path / "snapshots" / "v2" / "model.gguf"
    link2.parent.mkdir(parents=True)
    try:
        os.symlink(blob, link1)
        os.symlink(blob, link2)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this machine")
    res = scan([str(tmp_path)], with_metadata=False)
    assert len(res.models) == 1                     # both links resolve to one blob
    assert res.models[0].size_bytes == 128


def test_multipart_grouping(tmp_path):
    d = tmp_path / "snap"
    _touch(d / "big-00001-of-00003.gguf", size=10)
    _touch(d / "big-00002-of-00003.gguf", size=10)
    _touch(d / "big-00003-of-00003.gguf", size=10)
    res = scan([str(tmp_path)], with_metadata=False)
    assert len(res.models) == 1
    m = res.models[0]
    assert m.name == "big-00001-of-00003.gguf"      # part 1 is the model path
    assert len(m.parts) == 3
    assert m.size_bytes == 30                        # sizes summed


def test_missing_root_warns(tmp_path):
    res = scan([str(tmp_path / "nope")], with_metadata=False)
    assert res.models == []
    assert res.warnings
