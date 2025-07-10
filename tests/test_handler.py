from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.test_model
@pytest.mark.asyncio
async def test_handler_pipe(handler):
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

        result = await handler.pipe(test_data)

        assert "result" in result, f"missing 'result' key for sample {idx}"
        assert isinstance(result["result"], list), f"result is not a list for sample {idx}"

        print(f"IDX {idx}: {result['result']}")
