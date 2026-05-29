import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import FlagSpec, GgufMetadata, GpuInfo, HardwareInfo


@pytest.fixture
def client(isolated_data):
    # ensure default settings exist in the isolated data dir
    from app.core import store
    store.load_settings()
    return TestClient(app)


def test_configs_crud(client):
    r = client.post("/api/configs", json={"name": "c1", "model_display_path": "M", "flags": {"c": "100"}})
    assert r.status_code == 200
    cid = r.json()["id"]

    assert client.get("/api/configs").json()[0]["name"] == "c1"

    r = client.put(f"/api/configs/{cid}", json={"flags": {"c": "200"}})
    assert r.json()["flags"]["c"] == "200"

    assert client.delete(f"/api/configs/{cid}").status_code == 200
    assert client.get(f"/api/configs/{cid}").status_code == 404


def test_preview_and_export_roundtrip(client, tmp_path):
    client.post("/api/configs", json={"name": "c1", "model_display_path": "G:\\m.gguf", "flags": {"c": "16384"}})
    preview = client.get("/api/configs/preview")
    assert preview.status_code == 200
    assert "[c1]" in preview.text
    assert "model = G:\\m.gguf" in preview.text

    out = tmp_path / "out.ini"
    r = client.post("/api/configs/export", json={"output_path": str(out)})
    assert r.status_code == 200
    assert out.exists()

    # re-import yields the same entry
    imported = client.post("/api/configs/import", json={"text": out.read_text(), "replace": True})
    assert imported.json()[0]["name"] == "c1"


def test_export_missing_model_is_400(client):
    client.post("/api/configs", json={"name": "bad", "model_display_path": "", "flags": {}})
    r = client.post("/api/configs/export", json={"output_path": "x.ini"})
    assert r.status_code == 400


def test_models_scan_temp_root(client, tmp_path, monkeypatch):
    from app.core import store
    (tmp_path / "snap").mkdir()
    (tmp_path / "snap" / "model.gguf").write_bytes(b"\x00" * 32)
    s = store.load_settings()
    s.scan_roots = [str(tmp_path)]
    store.save_settings(s)

    r = client.post("/api/models/scan")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    assert "model.gguf" in names


def test_flags_endpoint_stubbed(client, monkeypatch):
    fake = [FlagSpec(canonical_key="gpu-layers", aliases=["-ngl"], value_type="int")]
    monkeypatch.setattr("app.api.routes_flags.get_flags", lambda exe, force_refresh=False: fake)
    r = client.get("/api/flags")
    assert r.status_code == 200
    assert r.json()[0]["canonical_key"] == "gpu-layers"


def test_hardware_endpoint_mocked(client, monkeypatch):
    hw = HardwareInfo(gpus=[GpuInfo(index=0, name="Fake", total_mib=16000, free_mib=14000)],
                      vram_source="nvidia-smi", ram_total_mib=64000, ram_available_mib=32000)
    monkeypatch.setattr("app.api.routes_hardware.detect", lambda: hw)
    r = client.get("/api/hardware")
    assert r.json()["gpus"][0]["total_mib"] == 16000


def test_suggest_endpoint_stubbed(client, monkeypatch):
    meta = GgufMetadata(architecture="qwen3", n_layers=32, context_length=32768,
                        n_head=32, n_head_kv=8, head_dim=128, file_size_bytes=8 * 1024 * 1024 * 1024)
    hw = HardwareInfo(gpus=[GpuInfo(index=0, total_mib=16000, free_mib=14000)], vram_source="nvidia-smi")
    monkeypatch.setattr("app.api.routes_suggest.read_gguf_metadata", lambda p: meta)
    monkeypatch.setattr("app.api.routes_suggest.detect", lambda: hw)
    r = client.post("/api/suggest", json={"model_path": "x.gguf", "context": 8192,
                                          "ctk": "f16", "ctv": "f16", "vram_budget_mode": "free"})
    assert r.status_code == 200
    body = r.json()
    assert "ngl" in body["explicit"]
    assert body["fit"]["fit"] == "on"


def test_settings_get_put(client):
    s = client.get("/api/settings").json()
    s["compute_reserve_mib"] = 2048
    r = client.put("/api/settings", json=s)
    assert r.json()["compute_reserve_mib"] == 2048
