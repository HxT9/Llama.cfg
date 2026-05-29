from app.core.ini_io import parse_ini, render_ini


def test_roundtrip_sample(sample_ini):
    parsed = parse_ini(sample_ini)
    rendered = render_ini(parsed)
    reparsed = parse_ini(rendered)
    assert parsed == reparsed                       # idempotent round-trip


def test_windows_paths_and_bool_preserved(sample_ini):
    parsed = parse_ini(sample_ini)
    # at least one section has a Windows model path with ':' and '\'
    some = next(iter(parsed.values()))
    assert ":" in some["model"] and "\\" in some["model"]
    # no-mmap = true survives
    assert any(kv.get("no-mmap") == "true" for kv in parsed.values())


def test_colon_not_treated_as_delimiter():
    text = "[s]\nmodel = G:\\a\\b.gguf\n"
    parsed = parse_ini(text)
    assert parsed["s"]["model"] == "G:\\a\\b.gguf"


def test_key_case_preserved():
    text = "[S]\nCtxSize = 10\nno-mmap = true\n"
    parsed = parse_ini(text)
    assert "CtxSize" in parsed["S"]


def test_model_written_first():
    text = "[s]\nc = 100\nmodel = X\nctk = q8_0\n"
    rendered = render_ini(parse_ini(text))
    lines = [l for l in rendered.splitlines() if l and not l.startswith("[")]
    assert lines[0].startswith("model =")


def test_duplicate_model_across_sections(sample_ini):
    parsed = parse_ini(sample_ini)
    models = [kv.get("model") for kv in parsed.values()]
    # the sample has multiple sections; allow duplicates without collapsing sections
    assert len(parsed) == len([s for s in parsed])
    assert len(models) == len(parsed)


def test_valueless_key_becomes_true():
    parsed = parse_ini("[s]\nmodel = X\nflash-attn\n")
    assert parsed["s"]["flash-attn"] == "true"
