import pytest

from app.core import store


def test_first_run_defaults(isolated_data):
    s = store.load_settings()
    assert s.scan_roots
    assert s.output_ini_path
    assert (isolated_data / "settings.json").exists()


def test_crud(isolated_data):
    e = store.create_entry("cfg1", model_display_path="M", flags={"c": "100"})
    assert store.get_entry(e.id).name == "cfg1"
    store.update_entry(e.id, {"flags": {"c": "200", "ctk": "q8_0"}})
    assert store.get_entry(e.id).flags["c"] == "200"
    assert store.delete_entry(e.id) is True
    assert store.get_entry(e.id) is None


def test_multiple_configs_per_model(isolated_data):
    store.create_entry("a", model_display_path="SAME", flags={"ctk": "q8_0"})
    store.create_entry("b", model_display_path="SAME", flags={"ctk": "f16"})
    entries = store.load_entries()
    assert len([e for e in entries if e.model_display_path == "SAME"]) == 2


def test_export_requires_model(isolated_data):
    store.create_entry("nomodel", model_display_path="", flags={})
    with pytest.raises(ValueError):
        store.entries_to_ini_text(store.load_entries())


def test_export_rejects_duplicate_section(isolated_data):
    store.create_entry("dup", model_display_path="X")
    store.create_entry("dup", model_display_path="Y")
    with pytest.raises(ValueError):
        store.entries_to_ini_text(store.load_entries())


def test_export_import_roundtrip(isolated_data, tmp_path):
    store.create_entry("c1", model_display_path="G:\\m1.gguf",
                       flags={"c": "16384", "no-mmap": "true"}, mmproj="G:\\mmproj.gguf")
    store.create_entry("c2", model_display_path="G:\\m2.gguf", flags={"ctk": "q8_0"})
    out = tmp_path / "out.ini"
    text = store.export_store_to_ini(str(out))
    assert out.exists()

    imported = store.import_ini_to_store(text, replace=True)
    by_name = {e.name: e for e in imported}
    assert by_name["c1"].model_display_path == "G:\\m1.gguf"
    assert by_name["c1"].mmproj == "G:\\mmproj.gguf"
    assert by_name["c1"].flags.get("c") == "16384"
    assert by_name["c2"].flags.get("ctk") == "q8_0"
