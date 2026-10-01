import socket

import matplotlib
import pytest
import requests

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Unit tests must not open sockets or perform HTTP."""

    def explode(*args, **kwargs):
        raise AssertionError("Network access is disabled in unit tests")

    monkeypatch.setattr(socket, "create_connection", explode)
    monkeypatch.setattr(requests.sessions.Session, "request", explode)
