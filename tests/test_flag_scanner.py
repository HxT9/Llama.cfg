from app.core.flag_scanner import parse_help_text


def test_parses_many_flags(help_text):
    flags = parse_help_text(help_text)
    assert len(flags) > 150
    by_key = {f.canonical_key: f for f in flags}
    assert "gpu-layers" in by_key
    assert "ctx-size" in by_key


def test_gpu_layers_aliases(help_text):
    by_key = {f.canonical_key: f for f in parse_help_text(help_text)}
    f = by_key["gpu-layers"]
    assert set(f.aliases) == {"-ngl", "--gpu-layers", "--n-gpu-layers"}
    assert f.value_type == "int"
    assert f.takes_value is True
    assert f.env == "LLAMA_ARG_N_GPU_LAYERS"


def test_ctx_size_is_int(help_text):
    f = {x.canonical_key: x for x in parse_help_text(help_text)}["ctx-size"]
    assert f.value_type == "int"
    assert "-c" in f.aliases


def test_cache_type_enum_from_allowed_values(help_text):
    f = {x.canonical_key: x for x in parse_help_text(help_text)}["cache-type-k"]
    assert f.value_type == "enum"
    assert "q8_0" in f.enum_values
    assert "f16" in f.enum_values


def test_cpu_moe_is_valueless_bool(help_text):
    f = {x.canonical_key: x for x in parse_help_text(help_text)}["cpu-moe"]
    assert f.takes_value is False
    assert f.value_type == "bool"
    assert "-cmoe" in f.aliases


def test_fit_tristate(help_text):
    f = {x.canonical_key: x for x in parse_help_text(help_text)}["fit"]
    assert f.value_type == "tristate"
    assert set(f.enum_values) == {"on", "off"}


def test_spec_type_bare_comma_enum(help_text):
    f = {x.canonical_key: x for x in parse_help_text(help_text)}["spec-type"]
    assert f.value_type == "enum"
    assert "draft-mtp" in f.enum_values


def test_mmap_toggle_is_bool(help_text):
    f = {x.canonical_key: x for x in parse_help_text(help_text)}["mmap"]
    assert f.value_type == "bool"
    assert "--no-mmap" in f.aliases


def test_groups_captured(help_text):
    flags = parse_help_text(help_text)
    groups = {f.group for f in flags if f.group}
    assert any("common" in (g or "").lower() for g in groups)
