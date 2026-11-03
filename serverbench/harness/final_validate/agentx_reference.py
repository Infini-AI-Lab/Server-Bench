"""Score a post-run uniform sample of actual AgentX replies.

The plain checkpoint teacher-forces every output token under its exact timed
prompt and the candidate's preceding tokens. The only numerical gate is the
maximum token log-probability regret over the entire sample:
log P(reference argmax) - log P(candidate token). There is no all-request
replay, per-reply gate, or special treatment of cache collisions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "final_validate.agentx_reference_replay/3"
DEFAULT_SAMPLE_SIZE = 8
DEFAULT_MAX_TOKEN_LOGPROB_REGRET = 1.0
# FP32 logprob subtraction can turn an exact unit gap into 1.0000000298.
# This is arithmetic roundoff tolerance, not another fidelity threshold.
LOGPROB_COMPARISON_ATOL = 1e-6


def _stream_token_ids(record: dict[str, Any]) -> list[int]:
    """Read native IDs, never infer token boundaries from decoded SSE text."""
    ids: list[int] = []
    for response in record.get("responses") or []:
        for packet in response.get("packets") or []:
            try:
                event = json.loads(packet.get("value", ""))
            except (ValueError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            for choice in event.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                chunk = (choice.get("delta") or {}).get("token_ids")
                if chunk is None:
                    chunk = choice.get("token_ids")
                if chunk is not None:
                    if (not isinstance(chunk, list)
                            or any(type(token) is not int or token < 0 for token in chunk)):
                        raise ValueError("stream token_ids must be nonnegative integers")
                    ids.extend(chunk)
    return ids


def _identity(row: dict[str, Any]) -> str:
    return f"{row.get('conversation_id', '')}:{row.get('turn_index', '')}"


def _read_outputs(artifact_dir: Path) -> list[dict[str, Any]]:
    path = Path(artifact_dir) / "outputs.json"
    body = json.loads(path.read_text("utf-8"))
    rows = body.get("data") if isinstance(body, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} contains no data list")
    return [row for row in rows if isinstance(row, dict)]


def _raw_export(artifact_dir: Path) -> Path:
    found = sorted(Path(artifact_dir).rglob("profile_export_raw.jsonl"))
    if not found:
        raise FileNotFoundError(
            f"{artifact_dir} contains no profile_export_raw.jsonl")
    return found[-1]


def select_timed_requests(artifact_dir: Path, *, seed: int,
                          sample_size: int = DEFAULT_SAMPLE_SIZE,
                          request_ids: list[str] | None = None,
                          ) -> dict[str, Any]:
    """Uniformly sample committed replies and recover only their wire payloads.

    `request_ids` replaces the uniform draw with an explicit list -- the EOS
    gate's targeted audit of replies that look padded.
    """
    if sample_size < 1:
        raise ValueError("AgentX reference sample_size must be positive")
    outputs = [row for row in _read_outputs(artifact_dir)
               if row.get("x_request_id")]
    outputs.sort(key=lambda row: (_identity(row), str(row["x_request_id"])))
    result = {
        "schema": SCHEMA, "evaluable": False, "holds": False,
        "selection_seed": int(seed), "requested_sample_size": sample_size,
        "available_outputs": len(outputs), "items": [], "why": "",
        "selection": {"method": "uniform_without_replacement",
                      "when": "after the timed replay", "tokens": "all"},
    }
    if len(outputs) < sample_size:
        result["why"] = (f"the timed replay retained only {len(outputs)} "
                         f"replies; {sample_size} are required")
        return result
    if request_ids is not None:
        wanted = {str(r) for r in request_ids}
        chosen = [row for row in outputs if str(row["x_request_id"]) in wanted]
        result["selection"]["method"] = "targeted"
        if len(chosen) != len(wanted):
            result["why"] = (f"{len(wanted) - len(chosen)} targeted replies are "
                             "absent from the committed outputs")
            return result
    else:
        chosen = random.Random(int(seed)).sample(outputs, sample_size)
    request_ids = {str(row["x_request_id"]) for row in chosen}
    source = _raw_export(artifact_dir)
    payloads: dict[str, dict[str, Any]] = {}
    native_ids: dict[str, list[int]] = {}
    native_errors: dict[str, str] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            metadata = record.get("metadata") or {}
            request_id = str(metadata.get("x_request_id") or "")
            if (request_id not in request_ids
                    or metadata.get("benchmark_phase") != "profiling"):
                continue
            if isinstance(record.get("payload"), dict):
                payloads[request_id] = record["payload"]
                try:
                    native_ids[request_id] = _stream_token_ids(record)
                except ValueError as exc:
                    native_errors[request_id] = str(exc)
            if len(payloads) == len(request_ids):
                break
    errors: list[str] = []
    items: list[dict[str, Any]] = []
    for row in chosen:
        request_id = str(row["x_request_id"])
        payload = payloads.get(request_id)
        if payload is None:
            errors.append(f"{request_id}: missing timed wire payload")
            continue
        try:
            greedy = float(payload.get("temperature")) == 0.0
        except (TypeError, ValueError):
            greedy = False
        if not greedy:
            errors.append(f"{request_id}: timed payload must carry temperature=0")
        if payload.get("ignore_eos") is not True:
            errors.append(f"{request_id}: timed payload must carry ignore_eos=true")
        response_text = str(row.get("response_text") or "")
        if not response_text:
            errors.append(f"{request_id}: missing committed response text")
        tokens = native_ids.get(request_id) or []
        if not tokens:
            errors.append(f"{request_id}: " + native_errors.get(
                request_id, "missing native output token_ids; retokenizing "
                "response text is not a valid substitute"))
        wire = json.dumps(payload, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        items.append({
            "id": _identity(row), "x_request_id": request_id,
            "payload": payload,
            "payload_sha256": hashlib.sha256(wire).hexdigest(),
            "candidate_response_text": response_text,
            "candidate_tokens": tokens,
            "candidate_response_sha256": hashlib.sha256(
                response_text.encode("utf-8")).hexdigest(),
        })
    result.update(evaluable=not errors, items=items,
                  raw_export=str(source), why="; ".join(errors))
    return result


_TEMPLATE_FIELDS = (
    "model", "messages", "tools", "tool_choice", "parallel_tool_calls",
    "chat_template", "chat_template_kwargs", "add_generation_prompt",
    "continue_final_message",
)


def _post_json(url: str, path: str, body: dict[str, Any], *,
               timeout_s: float) -> dict[str, Any]:
    wire = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}{path}", data=wire, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(  # noqa: S310 - validator endpoint
            request, timeout=timeout_s) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"{path} returned no JSON object")
    if decoded.get("object") == "error" or decoded.get("error"):
        raise ValueError(str(decoded.get("message") or decoded.get("error")))
    return decoded


def score_reference_reply(url: str, payload: dict[str, Any], *,
                          candidate_tokens: list[int],
                          timeout_s: float = 900.0) -> dict[str, Any]:
    """One prefill scores the entire committed continuation; no generation."""
    started = time.monotonic()
    try:
        if not candidate_tokens:
            raise ValueError("candidate reply has no readable tokens")
        template = {key: copy.deepcopy(payload[key])
                    for key in _TEMPLATE_FIELDS if key in payload}
        tokenized = _post_json(url, "/tokenize", template, timeout_s=timeout_s)
        prompt_tokens = [int(value) for value in tokenized.get("tokens") or []]
        if not prompt_tokens:
            raise ValueError("/tokenize returned no prompt tokens")
        response = _post_json(url, "/generate", {
            "input_ids": prompt_tokens + candidate_tokens,
            "sampling_params": {"temperature": 0, "max_new_tokens": 0},
            "return_logprob": True, "top_logprobs_num": 1,
            # Include the preceding prompt token; its logprob is discarded.
            "logprob_start_len": len(prompt_tokens) - 1,
        }, timeout_s=timeout_s)
        meta = response.get("meta_info") or {}
        actual_rows = list(meta.get("input_token_logprobs") or [])[1:]
        top_rows = list(meta.get("input_top_logprobs") or [])[1:]
        if (len(actual_rows) != len(candidate_tokens)
                or len(top_rows) != len(candidate_tokens)):
            raise ValueError("reference logprob rows do not cover every output token")
        token_scores = []
        for position, (wanted, actual, choices) in enumerate(zip(
                candidate_tokens, actual_rows, top_rows, strict=True)):
            if int(actual[1]) != wanted:
                raise ValueError(f"reference token alignment failed at {position}")
            best = choices[0]
            actual_lp, best_lp = float(actual[0]), float(best[0])
            if not math.isfinite(actual_lp) or not math.isfinite(best_lp):
                raise ValueError(f"non-finite reference logprob at {position}")
            token_scores.append({
                "position": position, "token_id": wanted,
                "candidate_logprob": actual_lp, "top1_logprob": best_lp,
                "top1_token_id": int(best[1]),
                "logprob_regret": max(0.0, best_lp - actual_lp),
            })
        total = math.fsum(row["logprob_regret"] for row in token_scores)
        return {
            "answered": True, "teacher_forced": True,
            "prompt_tokens": len(prompt_tokens),
            "audited_tokens": len(token_scores),
            "total_logprob_regret": total,
            "mean_logprob_regret": total / len(token_scores),
            "max_logprob_regret": max(row["logprob_regret"] for row in token_scores),
            "token_scores": token_scores,
            "seconds": round(time.monotonic() - started, 3),
        }
    except (urllib.error.URLError, TimeoutError, OSError, ValueError,
            KeyError, IndexError, TypeError, OverflowError) as exc:
        return {
            "answered": False, "teacher_forced": True,
            "audited_tokens": 0, "token_scores": [],
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.monotonic() - started, 3),
        }


def compare(samples: dict[str, Any], references: list[dict[str, Any]], *,
            max_token_regret: float = DEFAULT_MAX_TOKEN_LOGPROB_REGRET,
            ) -> dict[str, Any]:
    """Apply one threshold to the worst native token across the full sample."""
    if not math.isfinite(max_token_regret) or max_token_regret < 0:
        raise ValueError("max_token_regret must be finite and nonnegative")
    if not samples.get("evaluable"):
        return dict(samples)
    by_id = {str(row.get("x_request_id")): row for row in references}
    results = []
    for item in samples["items"]:
        row = by_id.get(str(item["x_request_id"]), {})
        count = int(row.get("audited_tokens") or 0)
        total = row.get("total_logprob_regret")
        maximum = row.get("max_logprob_regret")
        valid = (bool(row.get("answered")) and count > 0
                 and count == item["candidate_token_count"]
                 and isinstance(total, (int, float))
                 and math.isfinite(total) and total >= 0
                 and isinstance(maximum, (int, float))
                 and math.isfinite(maximum) and 0 <= maximum <= total)
        results.append({
            **row, "id": item["id"], "x_request_id": item["x_request_id"],
            "payload_sha256": item["payload_sha256"],
            "candidate_response_sha256": item["candidate_response_sha256"],
            "candidate_token_count": item["candidate_token_count"],
            "reference_answered": valid,
        })
    complete = (len(results) == samples["requested_sample_size"]
                and all(row["reference_answered"] for row in results))
    valid_rows = [row for row in results if row["reference_answered"]]
    count = sum(row["audited_tokens"] for row in valid_rows)
    total = math.fsum(row["total_logprob_regret"] for row in valid_rows)
    mean = total / count if complete and count else None
    maximum = max((row["max_logprob_regret"] for row in valid_rows),
                  default=None) if complete else None
    holds = maximum is not None and (
        maximum <= max_token_regret or math.isclose(
            maximum, max_token_regret, rel_tol=0.0,
            abs_tol=LOGPROB_COMPARISON_ATOL))
    why = ""
    if not complete:
        why = (f"the plain reference scored only {len(valid_rows)}/"
               f"{samples['requested_sample_size']} selected replies completely")
    elif not holds:
        why = (f"sample max token logprob regret {maximum:.6g} exceeds "
               f"{max_token_regret:.6g}")
    return {
        "schema": SCHEMA, "available": True, "evaluable": complete,
        "holds": holds, "rule": "timed_agentx_vs_plain_checkpoint",
        "selection_seed": samples["selection_seed"],
        "selection": samples["selection"],
        "sample_size": samples["requested_sample_size"],
        "available_outputs": samples["available_outputs"],
        "audited_replies": len(valid_rows), "audited_tokens": count,
        "total_logprob_regret": total if complete else None,
        "mean_logprob_regret": mean,
        "max_logprob_regret": maximum,
        "max_token_logprob_regret": float(max_token_regret),
        "logprob_comparison_atol": LOGPROB_COMPARISON_ATOL,
        "results": results, "why": why,
    }


def replay_and_compare(url: str, artifact_dir: Path, *, tokenizer, seed: int,
                       sample_size: int = DEFAULT_SAMPLE_SIZE,
                       max_token_regret: float = DEFAULT_MAX_TOKEN_LOGPROB_REGRET,
                       destination: Path | None = None,
                       ) -> dict[str, Any]:
    """Sample, teacher-force each reply once, aggregate, and save evidence."""
    samples = select_timed_requests(
        artifact_dir, seed=seed, sample_size=sample_size)
    return replay_selected_and_compare(url, samples, tokenizer=tokenizer,
                                       max_token_regret=max_token_regret,
                                       destination=destination)


def replay_selected_and_compare(url: str, samples: dict[str, Any], *, tokenizer,
                                max_token_regret: float = DEFAULT_MAX_TOKEN_LOGPROB_REGRET,
                                destination: Path | None = None) -> dict[str, Any]:
    """Score an already selected sample, also used across isolated backends.

    Selection always happens after the timed replies are committed. This seam
    avoids transferring the entire raw traffic export into a remote oracle.
    """
    started = time.monotonic()
    references: list[dict[str, Any]] = []
    if samples.get("evaluable"):
        def replay(item: dict[str, Any]) -> dict[str, Any]:
            tokens = item["candidate_tokens"]
            item["candidate_token_count"] = len(tokens)
            try:
                decoded = tokenizer.decode(
                    tokens, skip_special_tokens=False,
                    clean_up_tokenization_spaces=False)
            except (ValueError, TypeError, IndexError, OverflowError) as exc:
                return {"x_request_id": item["x_request_id"], "answered": False,
                        "error": f"cannot decode native token_ids: {exc}"}
            if decoded != item["candidate_response_text"]:
                return {"x_request_id": item["x_request_id"], "answered": False,
                        "error": "native token_ids do not decode to the committed response text"}
            response = score_reference_reply(
                url, item["payload"], candidate_tokens=tokens)
            return {"x_request_id": item["x_request_id"], **response}

        references = [replay(item) for item in samples["items"]]
    result = compare(samples, references, max_token_regret=max_token_regret)
    result.update(
        reference_arm="sglang",
        reference_concurrency=1,
        reference_method="teacher_forced_all_output_tokens",
        seconds=round(time.monotonic() - started, 3))
    if destination is not None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=1, allow_nan=False)
                               + "\n", "utf-8")
    return result
