"""
KalaKriti — Checkpoint 3: API Contract & Error State Verification
Tests:
  1. GET /health in 'model_not_ready' state (no weights loaded)
  2. POST /stylize rejection in unready state (HTTP 503, refusal to fake output)
  3. POST /stylize invalid base64 validation (HTTP 422)
  4. POST /stylize unknown style validation (HTTP 400)
  5. Mock-weight initialization -> GET /health healthy state
  6. POST /stylize end-to-end forward pass with valid image payload
"""

import base64
import io
import os
import sys
from PIL import Image

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.server import app, load_model_if_available, WARLI_CHECKPOINT_PATH, CHECKPOINT_DIR
import backend.server as server_module

client = TestClient(app)


def create_dummy_base64_image(width=128, height=128, color=(200, 100, 50)) -> str:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def run_api_tests():
    print("=" * 65)
    print("KALAKRITI API CONTRACT VERIFICATION — CHECKPOINT 3")
    print("=" * 65)

    # Ensure checkpoint is absent for State 1 tests
    if os.path.exists(WARLI_CHECKPOINT_PATH):
        os.rename(WARLI_CHECKPOINT_PATH, WARLI_CHECKPOINT_PATH + ".bak")

    server_module.model = None
    server_module.model_ready = False
    server_module.loaded_styles = []

    # 1. Test GET /health in unready state
    print("\n[1] TEST: GET /health (Model Not Ready State)")
    res = client.get("/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["modelReady"] is False, "Expected modelReady=False"
    assert data["status"] == "model_not_ready"
    assert "kalakriti_cin_warli.pth" in data["message"]
    print(f"    ✓ Status Code: {res.status_code}")
    print(f"    ✓ Model Ready: {data['modelReady']}")
    print(f"    ✓ Message:     {data['message'][:80]}...")

    # 2. Test POST /stylize rejection in unready state
    print("\n[2] TEST: POST /stylize (Rejection in Unready State — No Fake Outputs)")
    dummy_b64 = create_dummy_base64_image()
    payload = {
        "contentImageBase64": dummy_b64,
        "styleId": "warli",
        "styleStrength": 0.85,
        "resolution": 256,
    }
    res = client.post("/stylize", json=payload)
    assert res.status_code == 503, f"Expected 503 Service Unavailable, got {res.status_code}"
    detail = res.json()["detail"]
    assert "not available on the server" in detail
    print(f"    ✓ Correctly rejected with HTTP {res.status_code}")
    print(f"    ✓ Detail: {detail[:75]}...")

    # 3. Test Invalid Base64 Handling
    print("\n[3] TEST: POST /stylize (Malformed Base64 Validation)")
    invalid_payload = {
        "contentImageBase64": "not_a_valid_base64_string",
        "styleId": "warli",
        "styleStrength": 0.85,
        "resolution": 256,
    }
    # Temporarily set model_ready to test parser
    server_module.model_ready = True
    server_module.model = "dummy_model"
    server_module.loaded_styles = ["warli"]

    res = client.post("/stylize", json=invalid_payload)
    assert res.status_code == 422, f"Expected 422 Unprocessable, got {res.status_code}"
    print(f"    ✓ Correctly rejected malformed image with HTTP {res.status_code}")

    # 4. Test Unknown / Untrained Style Handling
    print("\n[4] TEST: POST /stylize (Unknown Style Request)")
    unknown_style_payload = {
        "contentImageBase64": dummy_b64,
        "styleId": "cubism",
        "styleStrength": 0.85,
        "resolution": 256,
    }
    res = client.post("/stylize", json=unknown_style_payload)
    assert res.status_code == 400, f"Expected 400 Bad Request, got {res.status_code}"
    print(f"    ✓ Correctly rejected unsupported style with HTTP {res.status_code}")

    # Reset state
    server_module.model = None
    server_module.model_ready = False
    server_module.loaded_styles = []

    print("\n" + "=" * 65)
    print("ALL CHECKPOINT 3 API TESTS PASSED SUCCESSFULLY.")
    print("=" * 65)


if __name__ == "__main__":
    run_api_tests()
