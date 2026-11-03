"""runtime -- the three comparison frameworks, pinned in one place.

A baseline's version is part of what a measurement MEANS. "SGLang is faster than
your system" is a claim about a specific build, and a datasheet that cannot say
which build is a datasheet nobody can reproduce or dispute.

This map and `final_validate/arms.py::IMAGES` must agree (a test holds them
equal): the concurrency search that measures the bars reads this one, and the
scenario, verdict and Modal paths read the other. Each pin is a stable release
tag, never a floating tag or a release candidate, so a re-run reproduces the
build a number was measured on.

CHANGING A PIN INVALIDATES EVERY MEASUREMENT TAKEN UNDER THE OLD ONE.
That is not a reason to avoid changing it. It is a reason to change it
deliberately, re-measure, and never compare a number from one build against a
number from another.
"""

from __future__ import annotations

#: arm -> its pinned image and version, and why this build.
BASELINE_IMAGES: dict[str, dict[str, str]] = {
    "sglang": {
        "image": "lmsysorg/sglang:v0.5.19",
        "version": "0.5.19",
        "why": "a stable release tag, so every measurement names one build",
    },
    "vllm": {
        "image": "vllm/vllm-openai:v0.28.0",
        "version": "0.28.0",
        "why": ("a stable release recent enough to support draft-model speculative "
                "decoding, so an UNSUPPORTED finding is about vLLM, not the pin"),
    },
    "tensorrt-llm": {
        "image": "nvcr.io/nvidia/tensorrt-llm/release:1.2.1",
        "version": "1.2.1",
        "why": ("the latest stable release on NGC; a release candidate is a moving "
                "target with a version number"),
    },
}


def image_for(arm: str) -> str:
    """The pinned image for one arm, or a refusal naming the three that exist."""
    try:
        return BASELINE_IMAGES[arm]["image"]
    except KeyError:
        raise KeyError(
            f"{arm!r} is not a baseline this suite pins. The three are "
            f"{sorted(BASELINE_IMAGES)}") from None
