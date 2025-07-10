import base64
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.test_api
def test_health_check_status(client):
    response = client.get("/health_check")
    assert response.status_code == 200


@pytest.mark.test_api
def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the API"}


@pytest.mark.test_api
def test_predict_b64_endpoint_batch(client, mock_handler, mocker):
    mocker.patch("app.handler", mock_handler)

    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent
    test_csv_path = project_root / "test_data.csv"

    assert test_csv_path.exists(), f"{test_csv_path} не найден."

    # сколько строк прогнать через модель
    n = 3
    df = pd.read_csv(test_csv_path, nrows=n)

    for idx, row in df.iterrows():
        test_data = {
            "camera_regno": row["regno_recognize"],
            "nn_regno": row["afts_regno_ai"],
            "camera_score": float(row["recognition_accuracy"]),
            "nn_score": float(row["afts_regno_ai_score"]),
            "nn_sym_scores": row["afts_regno_ai_char_scores"],
            "nn_len_scores": row["afts_regno_ai_length_scores"],
            "camera_type": row["camera_type"],
            "camera_class": row["camera_class"],
            "time_check": row["time_check"],
            "direction": int(row["direction"]),
        }

        encoded_data = base64.b64encode(json.dumps(test_data).encode("utf-8"))

        response = client.post("/predict_b64", data=encoded_data)

        assert response.status_code == 200

        decoded_response = base64.b64decode(response.content).decode("utf-8")
        response_json = json.loads(decoded_response)

        print(f"IDX {idx}: {response_json['result']}")

        assert "result" in response_json
        assert isinstance(response_json["result"], list)
