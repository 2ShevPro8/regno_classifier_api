from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app import app
from handler import Handler


@pytest.fixture
def mock_handler(mocker):
    mock_handler = mocker.patch("handler.Handler", autospec=True)
    mock_handler_instance = mock_handler.return_value

    mock_handler_instance.pipe = AsyncMock(
        return_value={"result": [0.98, 0.02]}
    )
    return mock_handler_instance


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def handler():
    return Handler()
