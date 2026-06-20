from pathlib import Path

from xdiff.core import dask_runtime
from xdiff.model.request import CompareMode, CompareRequest


def make_local_request():
    return CompareRequest(
        input_mode=CompareMode.DIRECTORIES,
        reference_path=Path("ref"),
        comparison_path=Path("cmp"),
        filter_name="*.nc",
        common_pattern=None,
        variables=None,
        dask_workers=2,
    )


def test_client_from_request_logs_local_scheduler_address(monkeypatch, caplog):
    events = []

    class FakeLocalCluster:
        def __init__(self, **kwargs):
            events.append(("cluster_kwargs", kwargs))
            self.scheduler_address = "tcp://127.0.0.1:8786"
            self.dashboard_link = "http://127.0.0.1:8787/status"

        def close(self):
            events.append("cluster_closed")

    class FakeClient:
        def __init__(self, cluster):
            events.append(("client_cluster", cluster.scheduler_address))
            self.cluster = cluster

        def close(self):
            events.append("client_closed")

    fake_distributed = type(
        "FakeDistributed",
        (),
        {
            "LocalCluster": FakeLocalCluster,
            "Client": FakeClient,
        },
    )

    monkeypatch.setattr(dask_runtime, "_load_distributed", lambda: fake_distributed)

    with caplog.at_level("WARNING", logger="xdiff"):
        with dask_runtime.client_from_request(make_local_request()) as client:
            assert isinstance(client, FakeClient)

    assert "Local Dask scheduler available at tcp://127.0.0.1:8786" in caplog.text
    assert "Local Dask dashboard available at http://127.0.0.1:8787/status" in caplog.text
    assert events == [
        (
            "cluster_kwargs",
            {
                "n_workers": 2,
                "threads_per_worker": 1,
                "processes": True,
                "dashboard_address": ":8787",
            },
        ),
        ("client_cluster", "tcp://127.0.0.1:8786"),
        "client_closed",
        "cluster_closed",
    ]
