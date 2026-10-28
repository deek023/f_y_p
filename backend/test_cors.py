"""
KalaKriti — CORS Verification Test
Validates:
  1. GET /health responds with Access-Control-Allow-Origin: *
  2. OPTIONS /stylize (preflight) responds with 200 OK and valid CORS headers
  3. POST /stylize responds with Access-Control-Allow-Origin: *
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.server import app

client = TestClient(app)

FRONTEND_ORIGIN = "https://ais-dev-ss6iv6ai6pz4r3gv4zxmvt-562269579696.asia-southeast1.run.app"


def test_cors():
    print("=" * 65)
    print("KALAKRITI CORS CONFIGURATION VERIFICATION")
    print("=" * 65)

    # 1. GET /health with Origin
    print(f"\n[1] Testing GET /health with Origin: {FRONTEND_ORIGIN}")
    res = client.get("/health", headers={"Origin": FRONTEND_ORIGIN})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "access-control-allow-origin" in res.headers, "Missing access-control-allow-origin header"
    assert res.headers["access-control-allow-origin"] == "*", f"Expected '*', got {res.headers['access-control-allow-origin']}"
    assert res.headers.get("access-control-allow-credentials") is None or res.headers.get("access-control-allow-credentials") == "false"
    print(f"    ✓ Status: {res.status_code}")
    print(f"    ✓ Access-Control-Allow-Origin: {res.headers['access-control-allow-origin']}")

    # 2. OPTIONS /stylize preflight check
    print(f"\n[2] Testing OPTIONS /stylize preflight request")
    res_options = client.options(
        "/stylize",
        headers={
            "Origin": FRONTEND_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert res_options.status_code == 200, f"Expected 200, got {res_options.status_code}"
    assert "access-control-allow-origin" in res_options.headers, "Missing allow-origin in OPTIONS"
    assert res_options.headers["access-control-allow-origin"] == "*"
    print(f"    ✓ Preflight Status: {res_options.status_code}")
    print(f"    ✓ Access-Control-Allow-Origin:  {res_options.headers['access-control-allow-origin']}")
    print(f"    ✓ Access-Control-Allow-Methods: {res_options.headers.get('access-control-allow-methods')}")
    print(f"    ✓ Access-Control-Allow-Headers: {res_options.headers.get('access-control-allow-headers')}")

    # 3. POST /stylize with Origin
    print(f"\n[3] Testing POST /stylize with Origin")
    res_post = client.post(
        "/stylize",
        json={"contentImageBase64": "invalid_base64", "styleId": "warli", "styleStrength": 0.8, "resolution": 512},
        headers={"Origin": FRONTEND_ORIGIN},
    )
    assert "access-control-allow-origin" in res_post.headers, "Missing allow-origin in POST"
    assert res_post.headers["access-control-allow-origin"] == "*"
    print(f"    ✓ Status: {res_post.status_code} (Rejection handled gracefully)")
    print(f"    ✓ Access-Control-Allow-Origin: {res_post.headers['access-control-allow-origin']}")

    print("\n" + "=" * 65)
    print("✓ ALL CORS CHECKS PASSED: Server is fully open for frontend origin.")
    print("=" * 65)


if __name__ == "__main__":
    test_cors()
