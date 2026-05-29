from app.core.suggester import MIB, suggest
from app.models import GgufMetadata


def make_meta(file_gib=8, layers=32, moe=False):
    return GgufMetadata(
        architecture="qwen3" + ("moe" if moe else ""),
        n_layers=layers,
        context_length=32768,
        n_embd=4096,
        n_head=32,
        n_head_kv=8,
        head_dim=128,
        expert_count=128 if moe else 0,
        expert_used_count=8 if moe else 0,
        file_size_bytes=int(file_gib * 1024 * MIB),
        is_moe=moe,
    )


def test_ngl_monotonic_in_budget():
    meta = make_meta(file_gib=8, layers=32)
    low = suggest(meta, vram_mib=4096, context=4096)
    high = suggest(meta, vram_mib=24576, context=4096)
    assert low.explicit["ngl"] <= high.explicit["ngl"]
    assert high.explicit["ngl"] >= low.explicit["ngl"]


def test_ngl_capped_at_offloadable():
    meta = make_meta(file_gib=2, layers=16)
    s = suggest(meta, vram_mib=40000, context=4096)
    assert s.explicit["ngl"] <= meta.n_layers + 1
    assert s.breakdown["fits_all_layers"] is True


def test_kv_math_matches_hand_computation():
    meta = make_meta(file_gib=8, layers=32)
    s = suggest(meta, vram_mib=24576, context=8192, ctk="f16", ctv="f16")
    ngl = s.explicit["ngl"]
    # 2 bytes (f16) for K + 2 for V per element
    expected = ngl * 8192 * meta.n_head_kv * meta.head_dim * (2.0 + 2.0) / MIB
    assert abs(s.breakdown["kv_total_mib_at_ctx"] - round(expected, 2)) < 1.0


def test_moe_uses_full_size_and_hints_cpu_moe():
    # huge MoE that cannot fully fit a small GPU
    meta = make_meta(file_gib=36, layers=48, moe=True)
    s = suggest(meta, vram_mib=16376, context=8192)
    assert s.breakdown["is_moe"] is True
    assert s.breakdown["fits_all_layers"] is False
    assert s.breakdown["moe_hint"]
    assert any("cpu-moe" in w for w in s.warnings)
    # per-layer cost derived from FULL file size (not reduced by active experts)
    assert s.breakdown["bytes_per_layer_mib"] > 0


def test_explicit_moe_sets_n_cpu_moe():
    # big MoE: non-expert weights + KV fit, but full model doesn't -> spill
    # some experts to CPU while keeping all layers on GPU.
    meta = make_meta(file_gib=36, layers=48, moe=True)
    s = suggest(meta, vram_mib=12288, context=4096, expert_fraction=0.85)
    # MoE offload drives placement via n-cpu-moe and leaves ngl unset
    assert "ngl" not in s.explicit
    assert "n-cpu-moe" in s.explicit
    assert "cpu-moe" not in s.explicit                     # never emit cpu-moe
    assert 0 < s.explicit["n-cpu-moe"] <= meta.n_layers
    assert s.breakdown["ngl"] == meta.n_layers + 1         # effective: all layers on GPU
    assert s.breakdown["moe_offload"]


def test_moe_spill_increases_as_vram_shrinks():
    meta = make_meta(file_gib=36, layers=48, moe=True)
    big = suggest(meta, vram_mib=14000, context=4096, expert_fraction=0.85)
    small = suggest(meta, vram_mib=9000, context=4096, expert_fraction=0.85)
    assert small.explicit.get("n-cpu-moe", 0) >= big.explicit.get("n-cpu-moe", 0)


def test_moe_extreme_low_vram_all_experts_or_hint():
    # at a tiny budget, every layer's experts spill (n-cpu-moe == n_layers)
    # or it genuinely can't fit (hint). Never cpu-moe.
    meta = make_meta(file_gib=36, layers=48, moe=True)
    s = suggest(meta, vram_mib=6500, context=4096, expert_fraction=0.90)
    assert "cpu-moe" not in s.explicit
    assert ("n-cpu-moe" in s.explicit) or s.breakdown["moe_hint"]
    if "n-cpu-moe" in s.explicit:
        assert s.explicit["n-cpu-moe"] <= meta.n_layers


def test_dense_model_never_gets_moe_offload():
    meta = make_meta(file_gib=20, layers=40, moe=False)
    s = suggest(meta, vram_mib=8192, context=4096, expert_fraction=None)
    assert "n-cpu-moe" not in s.explicit
    assert "cpu-moe" not in s.explicit


def test_fit_fields_populated():
    meta = make_meta()
    s = suggest(meta, vram_mib=16376, context=16384, ctk="q8_0", ctv="q8_0")
    assert s.fit["fit"] == "on"
    assert s.fit["fitc"] == 16384
    assert s.fit["fitt"] > 0
    assert s.fit["ctk"] == "q8_0" and s.fit["ctv"] == "q8_0"
    # dense model that fits: fit config carries no MoE offload
    assert "n-cpu-moe" not in s.fit and "cpu-moe" not in s.fit


def test_fit_does_not_pin_moe_offload():
    # --fit owns device placement; the fit config must not carry n-cpu-moe,
    # even though the explicit suggestion does.
    meta = make_meta(file_gib=36, layers=48, moe=True)
    s = suggest(meta, vram_mib=12288, context=4096, expert_fraction=0.85)
    assert s.fit["fit"] == "on"
    assert "n-cpu-moe" not in s.fit
    assert "cpu-moe" not in s.fit
    assert "n-cpu-moe" in s.explicit          # explicit still pins it


def test_quantized_cache_smaller_than_f16():
    meta = make_meta()
    f16 = suggest(meta, vram_mib=8192, context=8192, ctk="f16", ctv="f16")
    q8 = suggest(meta, vram_mib=8192, context=8192, ctk="q8_0", ctv="q8_0")
    # smaller KV per token -> can offload at least as many layers
    assert q8.explicit["ngl"] >= f16.explicit["ngl"]


def test_mmproj_reduces_budget_and_ngl():
    meta = make_meta(file_gib=8, layers=32)
    base = suggest(meta, vram_mib=10000, context=4096)
    withmm = suggest(meta, vram_mib=10000, context=4096, mmproj_bytes=2 * 1024 * MIB)
    # accounting for a 2 GiB projector can only lower (or equal) the offload count
    assert withmm.explicit["ngl"] <= base.explicit["ngl"]
    assert withmm.breakdown["mmproj_mib"] == 2048
    # projector is included in the estimated VRAM usage
    assert withmm.breakdown["estimated_vram_used_mib"] >= 2048


def test_missing_metadata_returns_warning():
    s = suggest(GgufMetadata(file_size_bytes=0), vram_mib=8192)
    assert s.warnings
    assert s.explicit == {}
