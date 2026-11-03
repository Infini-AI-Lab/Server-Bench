"""The deploy scripts call what they name, with flags it accepts."""

from __future__ import annotations

import re
from pathlib import Path

from paths import SERVERBENCH

DEPLOY = SERVERBENCH / "scripts" / "deploy"


# --- the entry points must pass flags the scripts they call actually accept ---
#
# A step that passes a flag its target does not accept fails on argument
# parsing before it reaches a GPU. From the outside, a pipeline whose step dies
# that way is indistinguishable from one nobody has run lately.


def _shell_invocations(script: str) -> list[tuple[str, list[str]]]:
    """Every `<something>.py ... --flags` call in a shell script.

    Line continuations are joined first, because a flag on the far side of a
    backslash is still a flag being passed.
    """
    import re

    joined = re.sub(r"\\\s*\n\s*", " ", script)
    found = []
    for line in joined.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # `scripts/` groups its programs into folders, so the name alone does
        # not locate one. Prefer the path the invocation actually writes --
        # falling back to the bare name would silently resolve nothing and turn
        # this check green by never running it.
        match = (re.search(r"scripts/((?:[a-z_]+/)*[a-z_]+)\.py\b(.*)$", stripped)
                 or re.search(r"([a-z_]+)\.py\b(.*)$", stripped))
        if not match:
            continue
        flags = re.findall(r"(?<![\w-])--[a-z][a-z-]*", match.group(2))
        found.append((match.group(1) + ".py", sorted(set(flags))))
    return found


def _accepted_flags(path: Path) -> set[str] | None:
    """The long options a Python script's argparse declares."""
    import ast

    try:
        tree = ast.parse(path.read_text("utf-8"))
    except (OSError, SyntaxError):
        return None
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None)
        if name != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                flags.add(str(arg.value))
    return flags or None


def test_no_deploy_script_passes_a_flag_its_target_rejects():
    scripts = SERVERBENCH / "scripts"
    problems: list[str] = []
    for shell in sorted(DEPLOY.glob("*.sh")):
        for target, flags in _shell_invocations(shell.read_text("utf-8")):
            accepted = _accepted_flags(scripts / target)
            if accepted is None:          # not one of ours, or takes no options
                continue
            for flag in flags:
                if flag not in accepted:
                    problems.append(f"{shell.name} passes {flag} to {target}")
    assert not problems, (
        "these calls fail on argument parsing before they do any work: "
        + "; ".join(problems))


def test_the_mount_point_and_the_announced_model_path_agree():
    """Whatever a step mounts the model at, that is what it must announce.

    NOT "the path must be generic": a mount point can be a frozen contract with
    published versions that hardcode it in their own `server.py` and
    `serve.sh`, so this test checks agreement rather than taste.

    The real hazard is the two disagreeing -- mounting the checkpoint at one path
    and telling the agent's server it lives at another. That fails at model-load
    time with a message about a missing config.json, several minutes into a run.
    """
    problems = []
    for shell in sorted(DEPLOY.glob("*.sh")):
        text = re.sub(r"\\\s*\n\s*", " ", shell.read_text("utf-8"))
        mounts = set(re.findall(r'-v "\$MODEL_DIR:([^:"]+)', text))
        announced = set(re.findall(r"-e MODEL_PATH=([^\s\\]+)", text))
        if announced and mounts and not (announced & mounts):
            problems.append(
                f"{shell.name} mounts the model at {sorted(mounts)} and "
                f"announces MODEL_PATH={sorted(announced)}")
    assert not problems, "; ".join(problems)
