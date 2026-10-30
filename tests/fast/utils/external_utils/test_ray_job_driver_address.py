import json
import os
import shlex
import subprocess
import sys

import pytest

from miles.utils.external_utils import ray_job


@pytest.mark.parametrize("lifetime", ["independent", "launcher"])
@pytest.mark.parametrize("explicit", [False, True])
def test_driver_address_overrides_job_supervisor_discovery_without_changing_dashboard(monkeypatch, lifetime, explicit):
    dashboard = "http://127.0.0.1:8274"
    monkeypatch.setenv("RAY_ADDRESS", dashboard)
    monkeypatch.delenv("MILES_RAY_DRIVER_ADDRESS", raising=False)
    if explicit:
        monkeypatch.setenv("MILES_RAY_DRIVER_ADDRESS", "127.0.0.1:6384")
    captured = []

    def independent(command):
        argv = shlex.split(command)
        captured.append(shlex.join(argv[argv.index("--") + 1 :]))

    def launcher(**kwargs):
        assert kwargs["address"] == dashboard
        captured.append(kwargs["entrypoint"])

    monkeypatch.setattr(ray_job, "exec_command_cpu", independent)
    monkeypatch.setattr(ray_job, "_run_launcher_owned_job", launcher)
    entrypoint = shlex.join([sys.executable, "-c", "import os,json; print(json.dumps(os.environ['RAY_ADDRESS']))"])
    ray_job.run_ray_job(entrypoint=entrypoint, runtime_env={"env_vars": {}}, job_lifetime=lifetime)
    # Reproduce Ray Jobs replacing the environment with another discovered cluster.
    child = subprocess.run(
        ["bash", "-c", captured[0]],
        env={**os.environ, "RAY_ADDRESS": "127.0.0.1:18679"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(child.stdout) == ("127.0.0.1:6384" if explicit else "127.0.0.1:18679")
    assert os.environ["RAY_ADDRESS"] == dashboard
