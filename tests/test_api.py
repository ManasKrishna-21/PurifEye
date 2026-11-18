"""Unit tests for FastAPI view endpoints and responses."""

from unittest.mock import MagicMock
from starlette.requests import Request
import app.main as main_mod


def test_index_page_returns_200():
    req = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    response = main_mod.index(req)
    assert response.status_code == 200
    body = response.body.decode("utf-8")
    assert "PurifEye" in body
    assert "Live Detect" in body


def test_telemetry_endpoint_logic():
    mock_pipe = MagicMock()
    mock_pipe.fps = 24.5
    mock_pipe.inf_ms = 42.1
    mock_pipe.e2e_ms = 55.0
    mock_pipe.current_detections = [{"box": [1, 2, 3, 4]}]
    mock_pipe.unique_bottles = 7
    mock_pipe.dropped_frames = 1

    orig = main_mod.pipeline_instance
    main_mod.pipeline_instance = mock_pipe
    try:
        response = main_mod.get_telemetry()
        assert response.status_code == 200
        import json
        data = json.loads(response.body.decode("utf-8"))
        assert data["fps"] == 24.5
        assert data["inf_ms"] == 42.1
        assert data["active_detections"] == 1
        assert data["counted_bottles"] == 7
    finally:
        main_mod.pipeline_instance = orig
