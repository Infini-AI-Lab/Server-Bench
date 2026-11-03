"""Standing the reference framework up on the cards you actually asked for.

Each helper turns a failure that would otherwise look like something else (a
mangled launch line, a busy card) into one that names its cause.
"""

from __future__ import annotations

from pathlib import Path


def launch_line_from_file(path: str | Path) -> str:
    """A server launch line read from a FILE, comments and newlines stripped.

    Shell quoting can mangle a `--json-model-override-args` value so that the
    JSON reaches SGLang with literal backslashes and it dies on a
    `JSONDecodeError` that says nothing about serving. A file has no quoting
    to get wrong.
    """
    text = Path(path).expanduser().read_text("utf-8")
    return " ".join(line.strip() for line in text.splitlines()
                    if line.strip() and not line.strip().startswith("#"))


def gates_for(task_dir: Path):
    """The task's own frozen service levels. Read, never restated."""
    from final_validate import service_levels, task_spec
    return service_levels.read(
        {"service_levels": task_spec.contract(task_dir).get("service_levels")},
        task_id=task_dir.name)


def serve_extra_for(task_dir: Path, arm: str) -> str:
    """The arm's launch line as the task freezes it, admission limit filled."""
    from final_validate.scenario import baseline_serve_extra
    return baseline_serve_extra(task_dir, arm)


def docker_device_spec(gpu: str) -> str:
    """What `docker --gpus` wants, quoted. See `lease.docker_device_spec`."""
    from final_validate.lease import docker_device_spec as spec
    try:
        return spec(gpu)
    except ValueError:
        raise SystemExit("--gpu names no devices") from None


def wait_for_quiet_cards(device_spec: str, *, timeout_s: float = 900.0,
                         quiet_mib: int = 256) -> str:
    """"" when every target card is idle, else why not. See `lease.wait_for_quiet`."""
    from final_validate.lease import wait_for_quiet
    return wait_for_quiet(device_spec, timeout_s=timeout_s, quiet_mib=quiet_mib)
