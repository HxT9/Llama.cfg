import struct

import pytest

from app.core.gguf_reader import GGUF_MAGIC, read_gguf_metadata


def _write_minimal_gguf(path, arch="llama", kv_extra=None):
    """Build a tiny but valid GGUF (header + metadata only, 0 tensors)."""
    kv = {
        "general.architecture": ("str", arch),
        "general.name": ("str", "tiny"),
        f"{arch}.block_count": ("u32", 8),
        f"{arch}.context_length": ("u32", 4096),
        f"{arch}.embedding_length": ("u32", 512),
        f"{arch}.attention.head_count": ("u32", 8),
        f"{arch}.attention.head_count_kv": ("u32", 2),
    }
    if kv_extra:
        kv.update(kv_extra)

    def s(v):  # gguf string
        b = v.encode()
        return struct.pack("<Q", len(b)) + b

    buf = bytearray()
    buf += struct.pack("<I", GGUF_MAGIC)
    buf += struct.pack("<I", 3)          # version
    buf += struct.pack("<Q", 0)          # tensor_count
    buf += struct.pack("<Q", len(kv))    # kv_count
    type_id = {"u32": 4, "str": 8}
    for key, (t, val) in kv.items():
        buf += s(key)
        buf += struct.pack("<I", type_id[t])
        if t == "str":
            buf += s(val)
        elif t == "u32":
            buf += struct.pack("<I", val)
    path.write_bytes(bytes(buf))


def test_reads_dense_metadata(tmp_path):
    p = tmp_path / "tiny.gguf"
    _write_minimal_gguf(p)
    meta = read_gguf_metadata(str(p))
    assert meta.error is None
    assert meta.version == 3
    assert meta.architecture == "llama"
    assert meta.n_layers == 8
    assert meta.context_length == 4096
    assert meta.n_head == 8
    assert meta.n_head_kv == 2
    assert meta.head_dim == 64          # 512 / 8
    assert meta.is_moe is False
    assert meta.file_size_bytes > 0


def test_reads_moe_metadata(tmp_path):
    p = tmp_path / "moe.gguf"
    _write_minimal_gguf(p, arch="qwen3moe", kv_extra={
        "qwen3moe.expert_count": ("u32", 128),
        "qwen3moe.expert_used_count": ("u32", 8),
    })
    meta = read_gguf_metadata(str(p))
    assert meta.is_moe is True
    assert meta.expert_count == 128
    assert meta.expert_used_count == 8


def test_garbage_file_returns_error(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"not a gguf file at all")
    meta = read_gguf_metadata(str(p))
    assert meta.error is not None


@pytest.mark.requires_local
def test_real_small_gguf(small_gguf):
    meta = read_gguf_metadata(small_gguf)
    assert meta.error is None
    assert meta.version in (2, 3)
    assert meta.n_layers and meta.n_layers > 0
    assert meta.context_length and meta.context_length > 0


@pytest.mark.requires_local
def test_real_moe_gguf(moe_gguf):
    meta = read_gguf_metadata(moe_gguf)
    assert meta.error is None
    assert meta.expert_count > 1
    assert meta.is_moe is True
