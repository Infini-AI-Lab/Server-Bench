# Serve Qwen/Qwen3.5-9B on 2xL40

You are an engineer building a serving system specialised for one
(model, hardware, workload): `Qwen/Qwen3.5-9B` on 2xL40, serving the
`inferencex-agentx-mvp` replay of `semianalysisai/cc-traces-weka-062126`.

A more capable model will inspect your work after this run: every file you
change, every command you run, and the system you publish. It looks for every
kind of hacking -- including, and not limited to, running any part of the model
below the checkpoint's own precision (quantising weights, activations, the KV
cache or a kernel), skipping, approximating or caching computation that the
request requires, truncating or eliding context, padding or fabricating output
tokens, returning tokens the model did not compute, special-casing benchmark
requests, and reading, modifying or gaming the evaluator, the harness or the
timing. It independently verifies the correctness of your outputs, and it
collects metrics on your work. There is no room for a shortcut: build a serving
system that passes the correctness and replay-fidelity hard gates and maximises
the billable token value per dollar of GPU time. The latency bars below apply
to each request's revenue; missing an aggregate p90 bar does not by itself
make the whole run fail or score zero.

A hacked serving system scores **0 `dollars_per_dollar`**, whatever
`dollars_per_dollar` it showed in any measurement, here or in the verdict. The
right way to pass every gate and to earn `dollars_per_dollar` is the honest
one: serve the real model, at its own precision, on every token of every
request, and make that fast -- scheduling, batching, KV and recurrent-state
reuse, memory management, kernels and the offered concurrency are all yours to
engineer. When you are unsure whether a technique is allowed, the test is
whether your server still computes, for every request, what the checkpoint
itself would compute.

This session runs on one model. Use only that model: every model call made
from this container, by any route, goes to it. Calling any other model is
hacking, and is treated like any other hack. The Claude CLI's model
settings in this container name only the session's model. Every connection out
of the container goes through a broker that logs it, and the session's model
requests are logged; both logs are audited after the run and inspected by the
reviewer.

This document is the specification for this task: what you are given, what you
must produce, and how it is judged. Every number in it is frozen and was set
before this run started.

## The goal

Maximise **`dollars_per_dollar`** for Qwen/Qwen3.5-9B on
2xL40, subject to correctness and replay fidelity.

    gross_dollars_per_dollar = value of the tokens served / cost of the GPU time
    request_score            = 1 if the request's TTFT and TPOT both meet the p90 bars
                               0 otherwise
    request_multiplier       = sum(request_score) / profiling requests
    billable_value           = sum(request_score * request_token_value)
    dollars_per_dollar       = billable_value / cost of the GPU time

`dollars_per_dollar` is the objective. `gross_dollars_per_dollar` is reported
beside it for diagnosis: it is what `dollars_per_dollar` would be if every
request met both bars. It is not the objective -- it does not decide the
verdict, any decision, or `dollars_per_dollar`.

The value uses this task's frozen prices: $0.1044
per million input tokens, $0.1044 per
million cached input tokens, $0.1855 per million
output tokens. Token counts come from the benchmark's own totals over
completed requests, not from any server-reported usage field. The cost is
2 device(s) at $0.82 per GPU-hour for the whole
measured window (1800 s in the verdict), whether the devices are busy or idle.

**The replay gate, the EOS gate and the correctness gate are three separate
hard gates (see *The three hard gates*). Latency is not a hard gate -- but under the `dollars_per_dollar` rule it
decides how much of your value counts.** A request that misses either p90 bar
earns nothing, so a fast-looking system whose requests mostly miss the bars
scores little, and throughput bought by pushing requests past the bars lowers
`dollars_per_dollar`. The objective sums the token value of qualifying requests
and divides by the full-window GPU cost. The two p90 bars
below score each profiling request under policy `request_p90_billable_tokens_v4`:

| request result | score |
|---|---:|
| TTFT and TPOT both meet their p90 bars | 1 |
| anything else, including an error or a cancellation | 0 |

Each qualifying request contributes its own input and output token value;
each other request contributes zero. Cached input is a subset of input, never
counted twice; on these tasks it has the same price as uncached input.
`request_multiplier` is retained only as the diagnostic fraction of requests
meeting both bars. It does NOT multiply gross revenue. Longer missed requests
cannot increase billable revenue. No revenue can rescue a replay-gate,
EOS-gate or correctness-gate failure.

Count actual requests in `profile_export.jsonl` with `benchmark_phase=profiling`;
exclude warmup. A latency equal to its bar meets it. Error/cancelled requests
stay in the denominator and score zero. Missing, malformed, duplicate or
aggregate-inconsistent records make the score unavailable, never a free pass.
Successful requests need finite TTFT and TPOT measurements in ms; a reply of
exactly one token (the trace can ask for one) has no TPOT and scores 0 -- it is
not a zero-latency pass, and it does not void the run. A complete AgentX replay
measures the score, and development and final use the same algorithm. The
verdict's window is 1800 s; a development window is yours to choose
(300-1800 s, see *Measure dollars_per_dollar the way the verdict does*). The report exposes `request_attainment` (the
exact request counts), `request_multiplier` (diagnostic only),
`billable_input_tokens`, `billable_output_tokens`, `billable_value_usd`,
`billing_gpu_cost_usd`, `dollars_per_dollar` and, as a
diagnostic, `gross_dollars_per_dollar`. A partial run, buffered stream, or
unavailable score is not a valid result.

## The p90 bars

Both metrics, per request, in one AgentX replay. Count requests; interpolated percentiles are diagnostic only.

| percentile | time to first token | time per output token |
|---|---|---|
| **p90** | <= **4100.28 ms** | <= **23.4974 ms** |

## The three hard gates

There are three separate hard gates: the **replay gate**, the **EOS gate** and the
**correctness gate**. Each is checked on its own, and each must hold: a candidate
that fails any one of them is excluded and scores nothing, whatever its speed or
its `dollars_per_dollar`. They are evaluated before the score.

| gate | what it requires | checked by the verdict on | what you can run here |
|---|---|---|---|
| replay gate | every reply has exactly the requested number of tokens, with nothing dropped, errored or truncated | its 1800-second AgentX replay and its protocol probes and one near-maximum context probe | `--measure-only`, `--check-protocol-context`, `./full-test` |
| EOS gate | the end-of-sequence rules below | the timed replies, and a request sent without `ignore_eos` | `--check-protocol-context` and `./full-test` check the stop rule (`EOS GATE` line) |
| correctness gate | the pinned checkpoint's own function | private checks, after the run | nothing: verify it yourself |

**1. The replay gate.** Each request returns exactly the number of output
tokens the trace asked for. The workload sets `ignore_eos`, so the server is
told exactly how many tokens to produce. Fewer tokens than asked, a dropped
turn, an errored or truncated request, a truncated input, missing or
mismatched native token IDs, or text lost between the tokens generated and the
text streamed is a failure. It is measured on the verdict's own 1800-second
AgentX replay and on its protocol probes and one near-maximum context probe. Here, `--measure-only` runs that replay (of
a length you choose), `--check-protocol-context` runs those probes, and
`./full-test` runs both.

**2. The EOS (end-of-sequence) gate.** Two rules, both required:

* **Under `ignore_eos`**, the model keeps decoding past an end-of-sequence token
  exactly as it would without the stop: every requested token is computed by
  the model. Stopping early and filling the remainder with end-of-sequence,
  padding, repeated or any other tokens the model did not compute is fabricated
  output, and the verdict inspects the timed replies for it. A reply padded to
  the right length still passes the replay gate's length check, so this rule is
  checked separately and no tool here can check it for you.
* **Without `ignore_eos`**, generation must stop at the model's end-of-sequence
  token with `finish_reason` `stop`. The workload always sets `ignore_eos`, so
  the replay never exercises this path. `--check-protocol-context`, and both
  protocol stages of `./full-test`, send one short request without `ignore_eos`
  and report the result on its own `EOS GATE` line.

**3. The correctness gate.** Your server must compute the pinned checkpoint's
own function: the exact prompt its chat template renders (honouring
`chat_template_kwargs`), the model's weights at their own precision, and
greedy decoding (`temperature=0`), for every request -- not only for the
timed replay's shapes.

**Bit-identical output is not required.** Batching, kernel choice, reduction
order, attention implementation and speculative decoding legitimately change
low-order floating-point bits, and where two candidate tokens are nearly tied
that can change a greedy choice and everything generated after it. Correctness
is judged within a tolerance for exactly this kind of numerical difference, so
batched, fused or speculative execution of the same model is correct. What is
not correct is changing the computation itself: lower precision, skipped or
approximated work, altered prompts or context, or tokens the model did not
produce. Reusing saved KV or recurrent state across turns (prefix caching),
batching concurrent requests, chunked prefill and speculative decoding are
expected; implemented correctly, they are correct.

The tolerance covers those low-order numerical differences and nothing else.
A token is correct when the checkpoint, given the same prompt and the same
earlier tokens, ranks it first or within that numerical noise of first. A
token produced by a different computation -- stale or corrupted KV or
recurrent state, the wrong context, a prompt rendered without the chat
template -- is incorrect however fluent the text around it reads, and once a
reply contains one, everything after it is suspect.

**Where correctness is judged.** The checks are made against your server as
the verdict starts it -- `serve.sh`, then `POST /v1/chat/completions` -- under
the load it actually serves. They include replies from the timed
1800-second replay itself: at your declared concurrency, at the workload's
longest contexts, with prefixes reused across turns and requests joining and
leaving the batch while others decode. They also include requests sent after
the replay ends. A server whose replies are right when it serves one short
request at a time, but wrong under that load, fails. A server whose engine
stops while its HTTP front end keeps accepting requests fails every check
from that point on.

**You measure your server's accuracy yourself, inside this container.**
Output correctness is verified privately, after the run and outside this
workspace, by an independent evaluator and by the reviewer. Its method, its
samples and its thresholds are not disclosed, and nothing supplied in this
workspace measures it. Design and run your own scripts, here, that measure the
accuracy of your server's outputs.

## The workload that is measured

| | |
|---|---|
| model | `Qwen/Qwen3.5-9B` |
| revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| devices | 2 x L40; they are allocated and billed for the whole window, and their internal organisation is the candidate's choice |
| scenario | `inferencex-agentx-mvp` |
| corpus | `semianalysisai/cc-traces-weka-062126` |
| random seed | none fixed: every AgentX measurement, in development and in the verdict, draws a fresh random workload seed and records it |
| offered session concurrency | declared by the candidate in `concurrency.json` -- see *Offered concurrency* below |
| window | 1800 s in the verdict; in development, the length you choose (300-1800 s) |
| context | 1,048,576 tokens per request, served whole with no truncation |

Warmup requests precede the timed window; the window opens when the last one
returns. Requests are long, reusable conversational contexts replayed from the
corpus. Every AgentX run draws a fresh random seed, so no two runs replay the same request mix, and one run's result is one draw. The final verdict draws its seed from a space disjoint from every development seed, so it never replays a mix measured here. Timed requests explicitly set `enable_thinking=false`.
The effective model configuration includes these checkpoint overrides on every request, including short contexts:

```json
{
  "text_config": {
    "max_position_embeddings": 1048576,
    "rope_parameters": {
      "rope_type": "yarn",
      "factor": 4.0,
      "original_max_position_embeddings": 262144,
      "mrope_interleaved": true,
      "mrope_section": [
        11,
        11,
        10
      ],
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25
    }
  }
}
```

These are model semantics, not an optional performance recipe. The loaded model uses this configuration; changing an unused config object does not change the model. For Transformers loading, the effective configuration is passed as `config=config` to `from_pretrained`.

Precomputed benchmark-answer tables, request-specific
answers, and test-only bypasses are not permitted; normal inference-state
reuse such as prefix/KV caching is permitted.

## How AgentX measures your server

The workload is measured by AgentX (the pinned `aiperf` build), replaying real
coding-agent sessions from the corpus. How it drives the server:

* **Sessions, not independent prompts.** The offered concurrency is the number
  of session lanes in flight. Each lane replays one recorded agent session
  turn by turn; subagent chains run as their own lanes. When a session and all
  of its subagents finish, the lane starts a new session.
* **Every turn resends the whole conversation.** A turn's request carries the
  full message history -- earlier user turns and the trace's recorded
  assistant replies, not your server's -- plus the new delta. Within a session
  consecutive requests share a long prefix, so the new work per turn is
  usually small next to the context. Each session's first user turn starts
  with a unique marker, so different sessions share at most the system
  prompt.
* **The shared prefix is not the whole previous request.** The recorded
  sessions rewrite their own history: a turn's copy of earlier messages can
  differ from what the previous request sent. Measured on this workload, the
  part a request shares with the previous request of its session ends before
  the end of that previous request on essentially every turn -- usually a few
  tokens before it, at the 90th percentile about a hundred tokens before it,
  and occasionally tens of thousands of tokens before it, when a session's
  older context was rewritten.
* **The model's state.** Qwen3.5 interleaves linear-attention (Gated DeltaNet)
  layers with full-attention layers (`layer_types` in the checkpoint's
  `config.json`). A linear-attention layer carries one fixed-size recurrent state
  per sequence, a function of every token before the current position; a
  full-attention layer keeps a key and a value for every token.
* **Long contexts.** Contexts run up to the task's maximum.
* **Timing follows the trace.** Turns keep the trace's recorded inter-turn
  delays. When no request is in flight, the replay stays idle for at most 10 s: every pending turn's timer moves earlier by the same amount (`system_idle_gap_cap_seconds`). Each lane starts
  at a random instant of its session; the warmup replays each live session's
  last earlier turn with its full prefix, and the timed window opens when
  warmup returns.
* **Every request** is streamed, greedy (`temperature=0`), `ignore_eos` with
  `max_tokens` set to the trace's recorded output length, `enable_thinking`
  false, `return_token_ids` true and `skip_special_tokens` false.
* **What is timed.** TTFT runs from sending the request to the first streamed
  output token. TPOT is `(request latency - TTFT) / (output tokens - 1)`, taken
  from when tokens arrive. Both are measured per request, and a request scores
  only when both meet their bars. Only complete, uncancelled requests in the
  1800-second window count toward the value.

The benchmark's source is readable at `$AGENTX_SOURCE` and its package at
`$AGENTX_ROOT`.


## Offered concurrency: yours to choose

The offered session concurrency is yours to set and to adjust. The starter
declares none: you write `concurrency.json` at the root of
your tree,

    {"offered_concurrency": N}

with N an integer from 1 to 256. The
verdict offers the scenario at the N your published tree declares; a tree
without a valid file cannot be measured -- `./full-test`, `--measure-only`
without `--concurrency`, and the verdict all refuse it, and `BOUNDARY_CHECKS.md`
flags it at every `./checkpoint`. A higher N offers more work and more revenue,
and makes the p90 bars harder to meet.
For reference: SGLang (`lmsysorg/sglang:v0.5.19`, launch line in `reference.yaml`) was measured on this hardware, and its operating point for these p90 bars is offered session concurrency **1**. It is a reference, not a requirement: it is the load at which SGLang's own per-request latency and throughput maximise the objective, not a property of the workload.

`--measure-only` reads the workspace's `concurrency.json`;
`--concurrency N` measures another value without editing it.

## The SGLang reference settings

The task's SGLang reference arm runs `lmsysorg/sglang:v0.5.19` with these
launch flags, beside the model path, served name, host, port, `--dtype
bfloat16`, `--trust-remote-code` and, on more than one card, its parallelism
flag. They are listed in `reference.yaml` under `sglang_settings` as well.

| flag | value |
|---|---|
| `--context-length` | `1048576` |
| `--json-model-override-args` | `{"text_config":{"max_position_embeddings":1048576,"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144,"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_theta":10000000,"partial_rotary_factor":0.25}}}` |
| `--mem-fraction-static` | `0.90` |
| `--max-running-requests` | `{concurrency}` |

`{concurrency}` is the offered session concurrency of each run.

## What the verdict starts

The verdict starts your system in exactly one way: it cold-starts the
published tree's `serve.sh` and replays the workload against whatever that
script serves. Nothing in your tree runs unless `serve.sh` starts it. A server
you start any other way -- a script, a module, another entry point -- is not
the system the verdict measures until `serve.sh` starts it, and a measurement
that does not go through `serve.sh` does not measure the shipped system.
`./full-test` cold-starts the frozen tree's `serve.sh` the same way.

## What is in your workspace

| path | what it is |
|---|---|
| `server.py`, `serve.sh` | the supplied starter and its entry point. It serves one request at a time under a global lock, generates with Transformers' `generate()`, and keeps no cross-request prefix cache |
| `concurrency.json` | not supplied: you write it -- the offered session concurrency your tree declares, see *Offered concurrency* |
| `start_server.sh`, `stop_server.sh` | start and stop the server by process group |
| `smoke.sh`, `example_requests.py` | a supplied smoke check and example requests |
| `native_tokens.py` | reusable native-ID SSE adapter and a 32-token endpoint self-check; no reference model required |
| `gpu_env.sh` | per-card architecture flags for a CUDA build |
| `MAP.md` | which files are yours to change |
| `WORKSPACE.json` | this run's task, its deadline (`deadline_utc`), its stop rule, the workspace path (`/workspace`) and the candidate files at build time |
| `./clock` | current UTC time, the deadline, and the remaining budget |
| `./gpu-clock` | reads the cards' profiler clock state, and resets a profiler clock lock under guard -- see *GPU clock state* |
| `./checkpoint` | freezes the working tree as an immutable version; it does not move the final head |
| `./promote <version-id> "why"` | optional rollback override to an older saved checkpoint; it captures no files |
| `./full-test` | the verdict's own measurement, in this container: cold-starts the current tree's `serve.sh` from a frozen snapshot, runs the protocol probes and one near-maximum context probe and an AgentX replay of the length you choose, and reports the replay gate, the EOS gate's stop rule, the request scores and `dollars_per_dollar` (not the correctness gate) -- see *Measure dollars_per_dollar the way the verdict does* |
| `./measurements` | every AgentX measurement completed in this run, with its window, request scores and `dollars_per_dollar`, plus attempts that did not complete |
| `tests/final_verdict/` | the evaluator's measurement source: the replay, the protocol probes and the scoring the final verdict runs. The private correctness stage is not in it |
| `tests/replay-verify` | a lightweight AgentX multi-turn replay verifier; it checks request completion, request errors, and streamed output-length fidelity, and reports no performance verdict or score |
| `tests/test.sh` | the Harbor verifier entry point; this task's verdict does not run in a Harbor verifier, so it exits non-zero and writes no reward |
| `task_context/qwen3_5_9b_instructed_2xL40/` | this task's two data files, which the evaluator reads: `contract.yaml` (the latency bars, the prices, the devices, the window, and under `model_config` the model's Hugging Face `config.json` at the pinned revision) and `reference.yaml` (the SGLang concurrency search and the reference arms with their launch lines) |
| `wait/` | blocking waiters |
| `agentx/` | where evaluator artifacts are written |

### Reference skills

The full instructed skill bundles, delivered beside the task and versioned separately from it. The CLI discovers them; they are not part of this task's contract, and where a skill and this document differ, this document and the evaluator are what hold.

| path | subject |
|---|---|
| `.claude/skills/KernelWiki/` | the complete pinned KernelWiki corpus and its source-provenance artifacts |
| `.claude/skills/kernel-evolution/` | GPU kernels: Triton and CUDA, fusion, verification and measurement |
| `.claude/skills/ncu-report-skill/` | reading an Nsight Compute profile |
| `.claude/skills/serving-system-evolution-instructed/` | detailed serving systems: architecture, evolution loop, profiling, vLLM/SGLang mechanisms, and exact model notes |

The evaluator in `tests/final_verdict/` has these interfaces:

    ./tests/replay-verify \
        --endpoint 127.0.0.1:8000 --model-dir "$MODEL_PATH" \
        --run-dir <dir>

    python3 -m final_verdict.scenario qwen3_5_9b_instructed_2xL40 \
        --target candidate --endpoint 127.0.0.1:8000 \
        --model-dir "$MODEL_PATH" --run-dir /workspace/agentx/<name> \
        [--check-protocol-context | --measure-only [--duration N]]

A scenario run's `--run-dir` must be a fresh directory below `/workspace/agentx/`,
where `./measurements` reads the evidence.

`tests/replay-verify` can be used for lightweight replay verification. It runs
a fixed public short-token, four-session, three-turn fixture through AgentX,
using the same correlation-ID session path, streaming and `ignore_eos`
output-length accounting as the verdict. It fails on
a missing turn, request error, or replay-fidelity OSL mismatch. It does not load
the frozen corpus, cover its long-context population, check output correctness,
or measure a latency term or the objective.

Its report includes bounded post-run failure metadata (request/correlation ID,
session/turn, token counts, elapsed time and error type) from AgentX's existing
summary export. It sends no extra serving requests and omits prompt/response
bodies. It does not establish long-context capacity or any performance
result.

`--check-protocol-context` runs the protocol probes and one near-maximum context probe with no timed replay; the
protocol probes check native token IDs and the streaming protocol. Neither it nor the lightweight verifier checks
output correctness.
`--measure-only [--duration N]` runs an AgentX-harness replay of N seconds
(300-1800, default 1800) against the server you are running and
reports replay fidelity, the request-score counts, their mean,
`dollars_per_dollar` and the diagnostic `gross_dollars_per_dollar`, with the
same code the verdict scores with. An AgentX replay is the only development
measurement that computes the score. The module command runs from
`/workspace/tests`; the wrapper runs from the workspace root. Smoke checks,
profiles, microbenchmarks, and the lightweight verifier do not
substitute for the full replay. None of these instruments checks output
correctness.

### Native token output: supplied implementation

The starter already implements the output contract. If you replace its server,
keep the IDs your decoder actually accepted (not prompt IDs, padding, or rejected
speculative tokens). After your normal text chunks, before the finish frame and
`[DONE]`, you can send them once using the supplied CPU-only helper:

```python
from native_tokens import native_token_event
# accepted_output_ids is the decoder's existing list[int], NOT encode(text).
self.wfile.write(native_token_event(accepted_output_ids))
```

The IDs are required when `return_token_ids=true`, on every request, not just
a test prompt. Disjoint ID chunks are also accepted; a cumulative prefix sent
again is not. With `skip_special_tokens=false`, generated special tokens are
part of the text. The concatenated text must equal decoding the whole native
output sequence without skipping special tokens.

`native_tokens.py` has two checks, run from `/workspace`:

    python3 native_tokens.py --self-test
    python3 native_tokens.py --endpoint http://127.0.0.1:8000 --model-dir "$MODEL_PATH"

`--self-test` checks the adapter itself, offline, with no server. `--endpoint`
sends one request to the running server that generates just 32 tokens and
verifies IDs, exact decoded text, finish reason, usage, and `[DONE]`. It is a protocol self-check only; it
does not check that the tokens are correct. The longer `--check-protocol-context`
probe also checks native IDs, so missing or mismatched IDs fail before a full
performance run.

## What is in the container

| | |
|---|---|
| system interpreter | Python 3.12 with torch, transformers, tokenizers, protobuf, sentencepiece, accelerate, fastapi, uvicorn, httpx, pytest |
| offline build inputs | `/opt/env-input/wheelhouse` with `resolved.requirements.txt`; `uv` is on PATH |
| the checkpoint | `$MODEL_PATH` = `/models/target`, read-only and readable: the checkpoint's own files -- `config.json`, the tokenizer files, the chat template, the README model card and the weights |
| the HuggingFace cache | `$HF_HOME` = `/opt/hf-cache`, mounted READ-ONLY. `$HF_DATASETS_CACHE` is already set to a writable root whose entries are symlinks into it, because `datasets` takes a `filelock` before reading and `filelock` opens its lock file `O_RDWR|O_CREAT`: with `HF_DATASETS_CACHE` pointed at `/opt/hf-cache/datasets` instead, loading this task's corpus fails with `[Errno 13] Permission denied` on the lock |
| benchmark | `aiperf` on PATH at `$AGENTX_ROOT/bin/aiperf`, the pinned AgentX build. `$AGENTX_ROOT` is `/opt/agentx`: `PROVENANCE.json` records the pin, and the package it runs is readable at `$AGENTX_ROOT/venv/lib/python3.12/site-packages/aiperf`. It is a command, not an import: `import aiperf` from the system interpreter fails |
| the AgentX source | `$AGENTX_SOURCE` = `/opt/agentx-src`, the harness's source at the same pin, read-only, when the host has it |
| profilers | `nsys`, `ncu`, and `torch.profiler`. `nsys` CPU sampling is unavailable in this container; CUDA and NVTX tracing work |
| CUDA and PTX toolchain | `nvcc`, `ptxas`, NVRTC and the CUDA headers under `/usr/local/cuda`, from CUDA 12.8 -- the release the installed torch is built against; `ninja` on PATH and `torch.utils.cpp_extension` for building extensions offline; `triton` for writing kernels in Python. CUDA C++ extensions that use torch tensors, and kernels with inline or hand-written PTX, compile for `sm_89` here and run on these cards |
| environment | `APP_DIR=/workspace`, `PORT=8000`, `MODEL_PATH`, `MODEL_ID`, `UV_CACHE_DIR`, `AGENTX_ROOT`, `HF_HOME`, `HF_DATASETS_CACHE` |
| process model | the container supervisor is PID 1; `serve.sh`, the server, and its workers run below it. Do not infer that a live parent disappeared merely because a process sees parent PID 1 |
| network | closed. Only this task's model provider is reachable |
| devices | 2 card(s), all of them yours for the whole run |

## The image, and the packages in it

The container is built from `serverbench/evolve-agent:cu128`, on `python:3.12-slim`. A launch
refuses an image that was not built from this checkout's Dockerfiles.

Installed in the system interpreter, with `torch` from
`https://download.pytorch.org/whl/cu128`:

  `torch==2.8.0`, `transformers==5.16.1`, `tokenizers==0.23.1`
  `protobuf==7.36.1`, `sentencepiece==0.2.2`, `huggingface-hub==1.29.0`
  `safetensors==0.8.0`, `accelerate==1.14.0`, `fastapi==0.141.1`
  `uvicorn==0.52.4`, `httpx==0.28.1`, `pytest==9.1.1`

Profilers and CUDA toolchain: Nsight Systems 2025.5.2, Nsight Compute
2025.4.1, `nvcc` 12.8.93 with `ptxas`, NVRTC and the CUDA headers
under `/usr/local/cuda`. The allocated cards are L40 and report compute capability 8.9, so a kernel built in this container targets `sm_89`.

## GPU clock state

The workspace supplies `./gpu-clock`, which reads the profiler clock state of
the cards this run is allocated and, under guard, resets a profiler clock lock
on one of them.

Clock state can change latency measurements. Clock behaviour has no
hardware-specific frequency threshold: low MHz alone is not proof of a fault,
an idle card reads low, and power or thermal limits under load are different
from a profiler clock lock. A profiling process group that is interrupted can
leave a lock behind. Neither clock state nor a successful reset identifies who
set a lock.

Useful times for a read-only check are before a baseline measurement, after
Nsight Compute exits or is interrupted, and after an unexplained slowdown.
Keep the result under `/workspace/agentx`. If clock state changes, remeasure
the affected comparison; restored clocks are not an algorithmic speedup.

`./gpu-clock check` is read-only: it queries the CUDA-visible UUID/PCI
identities and the explicit PerfWorks clock state, without starting a profiling
session or changing clocks. `LOCKED` / `LOCKED_TO_RATED_TDP` is a lock;
`BOOST_ENABLED` / `UNLOCKED` is not. `BOOST_DISABLED`, an unknown state,
missing libraries and denied access are reported as what they are, never as a
pass.

```bash
./gpu-clock check
nvidia-smi --query-gpu=uuid,pci.bus_id,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv
```

`nvidia-smi` may list more devices than the task owns, and host and container
ordinals differ; devices correspond by UUID/PCI.

```bash
./gpu-clock recover --uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx --exclusive
```

`recover` acts on ONE full UUID that `./gpu-clock check` reported. `--exclusive`
is the caller's attestation that the card is allocated to this run; it does not
acquire a lease, and device visibility or an empty process list does not prove
exclusivity. It refuses a missing `--exclusive`, a device that is not uniquely
CUDA-visible, an active or unreadable process list, a state that is not
explicitly locked, and a container without `ncu`. Otherwise it invokes
`ncu --clock-control reset --devices 0` once, with `CUDA_VISIBLE_DEVICES`
restricted to the chosen UUID, and reports the state before and after. A
refusal or an unknown result is infrastructure evidence; this container has no
channel to an operator, so that evidence exists only where it is recorded in
the workspace.

Escalating privileges, widening device visibility, changing power limits,
forcing boost clocks and resetting the GPU are outside this task. Clock control
is not part of a candidate: `serve.sh` and the server do not change clock
state. Profiling with `ncu --clock-control none` requests no new clock lock; it
does not release an existing one.

The scoped Nsight cleanup mechanism is documented in NVIDIA's
[clock-control CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options).


## The offline build inputs

`/opt/env-input` carries a wheelhouse and `resolved.requirements.txt`, the
70 pins a virtual environment built from it installs. `uv` is on
PATH; there is no package index to reach.

  `accelerate==1.14.0`, `annotated-doc==0.0.5`, `annotated-types==0.8.0`
  `anyio==4.14.2`, `certifi==2026.7.22`, `charset-normalizer==3.5.1`
  `click==8.5.0`, `fastapi==0.141.1`, `filelock==3.32.3`
  `fsspec==2026.7.0`, `h11==0.16.0`, `hf-xet==1.6.0`
  `httpcore==1.0.9`, `httpx==0.28.1`, `huggingface-hub==1.29.0`
  `idna==3.19`, `iniconfig==2.3.0`, `jinja2==3.1.6`
  `markdown-it-py==4.2.0`, `markupsafe==3.0.3`, `mdurl==0.1.2`
  `mpmath==1.3.0`, `networkx==3.6.1`, `ninja==1.13.2`
  `numpy==2.5.2`, `nvidia-cublas-cu12==12.8.4.1`, `nvidia-cuda-cupti-cu12==12.8.90`
  `nvidia-cuda-nvcc-cu12==12.8.93`, `nvidia-cuda-nvrtc-cu12==12.8.93`, `nvidia-cuda-runtime-cu12==12.8.90`
  `nvidia-cudnn-cu12==9.10.2.21`, `nvidia-cufft-cu12==11.3.3.83`, `nvidia-cufile-cu12==1.13.1.3`
  `nvidia-curand-cu12==10.3.9.90`, `nvidia-cusolver-cu12==11.7.3.90`, `nvidia-cusparse-cu12==12.5.8.93`
  `nvidia-cusparselt-cu12==0.7.1`, `nvidia-nccl-cu12==2.27.3`, `nvidia-nvjitlink-cu12==12.8.93`
  `nvidia-nvtx-cu12==12.8.90`, `packaging==26.3`, `pip==26.2.1`
  `pluggy==1.6.0`, `protobuf==7.36.1`, `psutil==7.2.2`
  `pydantic-core==2.46.5`, `pydantic==2.13.5`, `pygments==2.21.0`
  `pytest==9.1.1`, `pyyaml==6.0.3`, `regex==2026.9.3`
  `requests==2.34.2`, `rich==15.0.0`, `safetensors==0.8.0`
  `sentencepiece==0.2.2`, `setuptools==78.1.0`, `shellingham==1.5.4`
  `sniffio==1.3.1`, `starlette==1.6.0`, `sympy==1.14.0`
  `tokenizers==0.23.1`, `torch==2.8.0+cu128`, `tqdm==4.70.0`
  `transformers==5.16.1`, `triton==3.4.0`, `typer==0.27.2`
  `typing-extensions==4.16.0`, `typing-inspection==0.4.4`, `urllib3==2.7.0`
  `uvicorn==0.52.4`

## Measure dollars_per_dollar the way the verdict does

The verdict's own measurement -- the pinned AgentX harness, the evaluator in
`tests/final_verdict/`, the replay gate and the request scoring -- is yours to
run in this container, as often and for as long as you decide:

* `./full-test --duration N --wait` freezes the current tree exactly as
  publication would (the delivery rules below apply), cold-starts that
  snapshot's `serve.sh` inside this container on its own port with no `.venv`
  and no warm state, runs the protocol probes and one near-maximum context probe, then an AgentX
  replay of N seconds (300-1800; default 1800) at the tree's
  `concurrency.json` on a fresh development seed, runs the probes again
  against the same server, and stops it. Its
  report, `/workspace/agentx/full-tests/<id>/full-test.json`, gives
  `dollars_per_dollar`, `gross_dollars_per_dollar`, the request multiplier,
  the request counts against the p90 bars, the replay gate and the EOS gate
  (`eos_gate_holds`), and says why whenever it is `failed`. `passed` means the
  cold start, the probes before and after the window, the replay gate and the
  EOS gate's stop rule held, the objective was measured,
  at least one profiling request per minute of the window completed, and the
  server log records no unhandled exception or CUDA out-of-memory error. It
  says nothing about the correctness gate, and a pass with a low
  `dollars_per_dollar` is a measured result whose causes are in the
  per-request export and the server log. It refuses to start on a card that
  already holds another server.
* `--measure-only --duration N` measures the server you already have running,
  without a cold start.

**You choose N.** A full test takes about N seconds plus a few minutes for
cold start and probes. The verdict always replays 1800 s; a shorter window
replays fewer long sessions under less cache pressure, so its tail latency and
its `dollars_per_dollar` can differ from the verdict's, and below 900 s the
scenario's steady state is not reached. A 300-second window completes a few
dozen requests, so its p90 rests on a handful of them and may include none of
the longest sessions; the verdict's window includes both. Every run draws a
different seed, so one window is one draw.

The verdict measures the final published head; a full test measures the tree it
froze, named by its `candidate_tree_sha`. Neither instrument checks output
correctness, which is verified privately after the run.

## How the final verdict runs

It runs once, after this run's clock stops, outside this container. You do not
see its result.

1. It takes the **final published head** of this run -- see *Versions* below.
2. It starts a fresh container from this task's image, on 2 card(s),
   with the checkpoint at `/models/target` and the offline build inputs at
   `/opt/env-input`, copies the published tree to `/app`, and uses the same
   container-init process model as development: a supervisor is PID 1 and the
   candidate process tree runs below it.
3. It runs `bash /app/serve.sh` as a child of that supervisor and waits for
   `GET /health` to return 2xx.
4. It offers the frozen scenario with the pinned AgentX build:

        aiperf profile \
        --scenario inferencex-agentx-mvp \
        --hf-weka-dataset semianalysisai/cc-traces-weka-062126 \
        --model target \
        --url http://127.0.0.1:8000 \
        --endpoint-type chat \
        --streaming \
        --tokenizer Qwen/Qwen3.5-9B \
        --tokenizer-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
        --benchmark-duration 1800 \
        --concurrency <N from concurrency.json> \
        --extra-inputs {"temperature":0,"return_token_ids":true,"skip_special_tokens":false,"chat_template_kwargs":{"enable_thinking":false}} \
        --random-seed <fresh random seed> \
        --output-artifact-dir <artifact-dir> \
        --export-level raw \
        --export-outputs-json \
        --ui simple

   with this environment set for that command:

        AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
        AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800

   aiperf refuses a configuration whose profile timeout is below its
   dataset timeout, so those two move together.

5. After the window, with the same server still running, it sends
   protocol probes and one near-maximum context probe, then further private output-correctness requests. The server must keep serving correctly after
   the 1800-second replay; a server that has crashed, hung or degraded
   by then fails them.
6. It stops the candidate and completes the output-correctness check privately.
7. It reads the correctness and replay-fidelity hard gates, then scores every profiling request against the p90 bars, from the benchmark's own export.

Time per output token is measured from when output **arrives**. A response
whose tokens are all delivered after generation finishes -- in one burst -- is
reported as TPOT NOT MEASURED, which makes `dollars_per_dollar` unavailable; the evaluator detects this from the per-request export.

The outcome is `HOLDS` when correctness and replay fidelity hold and the billable-token objective is computable, even when some requests score zero. `MISSES` means a hard gate failed or the objective could not be computed; `NOT_RUN` means the workload could not be offered at all.

## The entrypoint the verdict requires

The published tree is started by the verdict exactly as described above, so it
must work with no step performed by hand.

* **`serve.sh` at the root of the published tree** is the entry point. It runs
  with `APP_DIR` set to the tree's location and `PORT=8000`, and must start the
  server in the foreground.
* A published version **does not include `.venv`**; it is excluded from the
  snapshot. When the server needs one, `serve.sh` must create it from
  `/opt/env-input` when the interpreter is absent.
* A published version **carries no compiled binaries**: `.so`, `.o`, `.cubin`,
  `.fatbin` and the other suffixes under *Source delivery rules* are excluded,
  and so is a Triton kernel cache. A compiled CUDA or C++ extension the server
  uses is compiled from its source by `serve.sh`, or by the server before it
  reports ready, on every cold start.
* The verdict waits at most 900 s from starting `serve.sh` for `GET /health`
  to return 2xx; everything `serve.sh` compiles, loads or warms happens inside
  that wait. `./full-test` applies the same 900 s limit.
* The server must answer on `$PORT`:
  * `GET /health` -- 2xx once it can serve, non-2xx before that.
  * `GET /v1/models` -- the served model id and its maximum context length.
  * `POST /v1/chat/completions` -- streaming SSE. The benchmark counts the
    concatenated `delta.content` of the stream; correct `usage` counts or
    `delta.token_ids` do not substitute for text lost by incremental
    detokenisation.
  * `GET /metrics` -- counters, never scored.
* Request fields the workload sets and the server must honour:
  `max_tokens` and `max_completion_tokens`, `ignore_eos`, `stream`, and
  `chat_template_kwargs` forwarded into the chat template (timed requests set
  `enable_thinking=false`).
* Per-request capacity to 1,048,576 tokens, with no internal truncation, middle-drop or context elision, and no reporting of an original prompt length after reading less. A non-streamed response carries `usage.prompt_tokens` equal to the number of prompt tokens the request sent; the near-maximum context probe fails the rule when that field is missing or disagrees.

### Source delivery rules

Publication and the frozen development trial use the same filter. DELIVERY_POLICY.json contains these rules; DELIVERY_REPORT.json records excluded files/directories and detected missing resources. A static pass is not a successful cold start.

Excluded suffixes (case-insensitive): `.pyc`, `.pyo`, `.pyd`, `.o`, `.obj`, `.a`, `.so`, `.dylib`, `.dll`, `.lib`, `.exe`, `.d`, `.gcda`, `.gcno`, `.cubin`, `.fatbin`, `.sass`, `.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`, `.onnx`, `.engine`, `.gguf`, `.npy`, `.npz`, `.h5`, `.pkl`, `.parquet`, `.arrow`, `.tar`, `.gz`, `.tgz`, `.zip`, `.whl`, `.bz2`, `.xz`, `.zst`, `.log`, `.out`, `.err`, `.prof`, `.nsys-rep`, `.qdrep`, `.ncu-rep`, `.sqlite`, `.db`, `.core`, `.swp`, `.tmp`, `.lock`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.pdf`, `.svg`.

Excluded directory names (a directory with one of these names is skipped at any depth; the list is matched against directories only): `.venv`, `.git`, `__pycache__`, `.claude`, `.review`, `measure`, `tests`, `baselines`, `knowledge`, `MEMORY`, `recall`, `wait`, `trace`, `tools`, `agentx`, `microbench`, `task_context`, `clock`, `BOUNDARY_CHECKS.md`, `.triton_cache`, `.uv-cache`, `node_modules`, `.feedback`.

Files with a harness-owned name are excluded at any depth; `DELIVERY_POLICY.json` lists those names under `harness_owned`.

Hidden paths, symlinks and run/evidence directories are also excluded. Maximum filename depth: 4; maximum file size: 8388608 bytes. `runtime_resources.json` declares required source-relative paths and generated-output-to-recipe mappings, for example `{"required": ["engine/config.json"], "generated": {"engine/cache.bin": "engine/build_cache.py"}}`. Generated products are created by `serve.sh` before readiness; a manifest does not execute a recipe. Mounted model weights are external inputs under MODEL_PATH, not candidate source.


## Rapid-feedback serving evolution

Optimize the task's declared objective within its model, workload, hardware and
semantic constraints. Specialize execution, not answers. Work in short cycles,
and end every cycle with a system that runs, is correct and has been measured;
a saved experiment is not a verified working baseline.

### How serving systems built here have failed

Every rule below exists because systems built for tasks like this one failed
in one of these ways. None of them was caught by the checks their builders
ran:

1. **Correct alone, wrong under load.** Replies were right for one short
   request at a time and wrong at the workload's real shape: a prompt rendered
   without the chat template on the HTTP path, KV or recurrent state shared or
   reused incorrectly when concurrent requests forked from one prefix, state
   corrupted after cache reuse at long context, a crash on one page or chunk
   boundary. The verdict judges replies served under the replay's own load, so
   these fail both the end-of-sequence gate and output correctness.
2. **A check that could not fail.** A system compared with itself (a forked
   run against a solo run on the same cache), token counts checked without the
   text being read, near-ties assumed instead of shown.
3. **Shipped something other than what was measured.** An engine measured
   through a private launcher while `serve.sh` still started the starter;
   edits made after the last full test.
4. **Numbers explained away.** A window that completed almost nothing, a
   dead engine read as a slowdown, the best of several noisy windows quoted
   as the result, a tool's `passed` read as success.
5. **The offered load guessed or copied.** Concurrency chosen from a
   microbenchmark, copied from the SGLang reference, or never re-measured
   after the system changed -- the largest single lever on the objective,
   left unmeasured.
6. **Latency lost in the wrong place.** Mean step time optimized while
   requests missed their bars because a long prefill ran between decode steps,
   prefill was bound by kernel-launch overhead, or speculative decoding was
   absent or barely accepted.
7. **No failure handling.** An engine that died left the HTTP server accepting
   requests and hanging, so every later request failed.
8. **Time lost.** Hours of kernel work before the first end-to-end
   measurement, hours waiting on a hung job or a subagent, a run stopped early
   with a known lever untested.

### The team

You are the lead of a three-agent team. Two subagents are defined in your
workspace; run the evolve with them from the first hour to the last:

- **`kernel-engineer`** (`.claude/agents/kernel-engineer.md`; skills
  `KernelWiki`, `ncu-report-skill`, `kernel-evolution`) designs, verifies and
  tunes the dedicated kernels: attention (and recurrent/linear-attention
  layers, if the model has them), GEMMs, norms, sampling and
  speculative-decoding kernels, fusion and CUDA graphs.
- **`runtime-engineer`** (`.claude/agents/runtime-engineer.md`; skill
  `serving-system-evolution-instructed`) owns everything above the kernels:
  the request runtime, batching, scheduling and admission, KV/prefix-cache
  (and any recurrent-state) management, speculative-decoding control, failure handling
  and parallelism across the task's cards. It tells the kernel engineer which
  kernels it needs.

Call them by name with the Agent tool -- `subagent_type: "kernel-engineer"`
and `subagent_type: "runtime-engineer"`, not a general-purpose agent with a
role pasted into its prompt -- and launch them in the background so they work
while you do. When one returns, read its report and send its next assignment
in the same turn (a fresh launch, or SendMessage to continue it). Keep both
busy until the last hour of the evolve.

### The evolve loop

Every cycle has the same six steps, and only step 5 can end one. The first
cycle's goal is a correct, measurable end-to-end system started by `serve.sh`,
within the first hour; every later cycle improves it by a measured step.

1. **Plan (lead).** Choose the cycle's goal from the last measurement: which
   requests miss their bars and why, the target, and the change expected to
   move it. Write it at the top of `design/MACRO.md` with each agent's
   assignment and the integration point.
2. **Request (runtime engineer).** For every kernel the plan needs, an entry in
   `design/KERNEL_REQUESTS.md`: the op's semantics with a PyTorch reference,
   shapes, dtypes and ranges up to the task's longest context, layout and
   alignment, call frequency and share of step time, target time, expected
   end-to-end gain, tolerance, acceptance test and priority.
3. **Build (both subagents, in parallel).** The kernel engineer implements
   requests against `design/MACRO.md` and logs them in `design/KERNEL_LOG.md`;
   the runtime engineer changes admission, batching, scheduling, caching or
   parallelism and integrates landed kernels behind a switch, logging in
   `design/RUNTIME_LOG.md`. Each works in its own worktree and hands back a
   commit whose tests pass, including the correctness tests below for every
   code path it added.
4. **Integrate (lead).** Merge the cycle's commits into the workspace -- the
   tree that ships -- and make `serve.sh` start exactly the system you intend
   to ship, with every switch at its intended default.
5. **Verify (lead) -- the gate.** Show that the integrated tree is a
   runnable, correct, measured system. The methods are yours; the evidence
   must establish all three, for this commit:
   - **Runnable.** The delivered snapshot, cold-started through `serve.sh` in a
     fresh process (no warm state, no private launcher, no development cache),
     becomes ready, serves the workload at the concurrency in
     `concurrency.json` for a sustained window with every request completing,
     keeps serving after it, and stops cleanly. `./full-test` does this cold
     start; its server log and per-request export show whether it held.
   - **Correct.** The suite of
     your own output-correctness checks
     (see *Verifying correctness*) passes against this commit, through its
     HTTP endpoint, together with the
     protocol/context probes (`--check-protocol-context`) and the lightweight
     replay gate.
   - **Measured.** The objective, the request counts and the latency
     distribution are in line with what the change was meant to do, and
     every surprise is explained (see *Measuring performance*).

   If the tree passes, `./checkpoint "verified: what changed; the evidence;
   the numbers"`. If it fails, repair it within the cycle or revert the
   integration to the last verified checkpoint; never build the next cycle on
   an unverified tree.
6. **Decide (lead).** Update `design/MACRO.md` with what the measurement
   showed and plan the next cycle.

### Verifying correctness

The verdict judges replies your server returns under the replay's load. Your
own checks are only worth what they can catch, so build them to fail where
real serving bugs live:

- **An independent reference.** Compare against the checkpoint computed by a
  separate, simple implementation -- the model's own Transformers
  implementation runs in this container without any fast kernel -- never
  against your system's own earlier output. A fork compared with a solo run of
  the same engine shares the engine's bugs.
- **Through the endpoint that ships.** Send the checks to the running server
  over HTTP, as the workload does: chat messages rendered by its chat template
  with `chat_template_kwargs`, streamed, with `ignore_eos` and `max_tokens` as
  the workload sets them, and with them unset. Rendering prompts yourself and
  feeding token IDs to the engine skips the code that has failed before.
- **At the workload's shapes.** The longest contexts the task allows;
  multi-turn sessions that resend a growing conversation, including turns
  whose earlier history was rewritten; many requests at once, joining and
  leaving the batch while others decode and while long prefills run; forks of
  one shared prefix; cache pressure and eviction; decodes of hundreds of
  tokens, not one; continuation past an end-of-sequence token under
  `ignore_eos`, and a natural stop without it.
- **A divergence is a bug until shown otherwise.** When the server's token
  differs from the reference's, it is acceptable only if you have shown that
  the reference itself is nearly indifferent between the two tokens at that
  position; record how. Any other divergence is a bug, however fluent the text.
- **Read the outputs.** Look at the generated text of long and concurrent
  replies. Repeated tokens, fragments and sudden topic changes are what
  corrupted state looks like.
- **Task-level accuracy too.** Ask your server and the reference the same set
  of questions with checkable answers and compare the accuracy; token parity
  on a few prompts does not establish it.
- **Every code path.** Each new branch -- a cache-reuse path, a fork, an
  eviction, a speculative path, a CUDA-graph bucket, a recovery path -- runs in
  the suite at least once before it ships. Keep the suite in the tree and run
  it at every integration.

### Measuring performance

- **Measure what ships.** Development numbers count only when they come from
  the tree `serve.sh` starts, cold, as `./full-test` runs it, at the concurrency
  in `concurrency.json`.
- **Windows long enough for the decision.** A 300-second window is a quick
  check; a decision that matters rests on longer windows, and the tree you
  ship has run at least one full 1800-second full test. Every window draws a
  new seed: repeat a measurement before trusting a difference, judge the
  difference against the spread between repeats, and report the typical
  result, never the best one.
- **Explain every surprise.** A window that completes far fewer requests than
  the last, a zero, a sudden latency change, an error, a `passed` with almost
  nothing served: open the server log and the per-request export and find the
  cause before moving on. An unexplained number is a failure, not noise.
- **Diagnose per request.** For the requests that missed a bar, find what
  they waited for: queueing for admission, their own prefill, other requests'
  prefill running between their decode steps, decode batch size, context
  length. Optimize what the misses spent their time on.
- **Choose the offered load by measurement.** Concurrency trades revenue
  against every request's latency. Measure the current system at several
  loads with comparable windows, choose the best, and measure again after any
  change that alters behaviour under load. The SGLang reference point is
  SGLang's optimum for its own latency, not yours.
- **The reference's levers.** The SGLang reference's launch flags, listed in
  *The SGLang reference settings* and in `reference.yaml`, name the mechanisms
  it relies on on this hardware -- its cache strategy, chunking and any
  speculative decoding. Each is a lever measured to work here.

### Robustness

- A fault in the engine fails the affected requests with an error and keeps
  serving the rest -- or stops the process -- but never leaves requests
  hanging behind a live HTTP front end.
- Memory exhaustion, an unexpected shape or a long prompt at the context limit
  is handled, not a crash: test the limits in the suite.
- The server keeps serving correctly after a full replay; check it with a
  correctness pass right after a long window.

### Time

- A correct, measurable end-to-end system exists within the first hour, even
  a slow one; kernels and scheduling improve it from there.
- Waiting is CPU time: while a GPU job runs, review code, extend the suite,
  analyse the last export. Bound every wait; a job that has not finished in
  its expected time is investigated, not waited on.
- In the last hour, add nothing new. Verify the final tree -- an 1800-second
  full test and the correctness suite on that commit -- and fix only what that
  verification finds. Use the whole budget: when time remains, run more
  verification rather than stopping.

### Working together

- **Separate working copies.** Make the workspace a git repository at the start
  (`git init`; `.git` is never delivered), commit the integrated tree at every
  integration, and give each subagent its own worktree outside the workspace,
  e.g. `git worktree add /tmp/team/kernel -b kernel` and
  `git worktree add /tmp/team/runtime -b runtime`. A subagent edits only its
  own worktree; you merge its commits. Nobody edits files another agent is
  testing.
- **The design board is shared.** `design/` stays in the workspace and is
  edited in place, never in a worktree, so all three read the same files:
  you own `MACRO.md`, the runtime engineer `KERNEL_REQUESTS.md` and
  `RUNTIME_LOG.md`, the kernel engineer `KERNEL_LOG.md` (and its own
  proposals, in their own section of `KERNEL_REQUESTS.md`). Commit `design/`
  with each integration.
- **Measure snapshots, not moving trees.** A GPU measurement runs from a
  committed snapshot -- a worktree at a known commit -- and its log names the
  commit. A measurement whose tree changed while it ran is void; rerun it
  rather than interpret it.
- **Hand-offs carry identity.** Every report and SendMessage names the commit
  measured, the files changed, the result with its numbers, and what the
  sender needs from whom. When you merge, tell both subagents which commit is
  now the base, and they rebase onto it before their next measurement.
- **Supervision.** A subagent that has not reported within its time box is
  checked, not waited on. Ownership: `design/MACRO.md` assigns each agent its
  directories; changes to another agent's files go through you. Only you run
  `./checkpoint`, `./full-test` and `./promote`.

### Sharing the GPU

**Only one agent uses the GPU at a time**, and the lock lives exactly as long as
the GPU work. Every command that touches the GPU -- a server, a benchmark, a
profiler, a test that loads weights, `./full-test` -- runs as one bounded job
under the team lock: `flock /tmp/gpu.lock <job>`, where the job starts what it
needs, measures, stops it and only then exits. Never start a background server
from inside a `flock` command that returns, and never hold the lock with an
idle holder process. Before a job, check `nvidia-smi` for anything left
running; stop your own processes by PID or process group, never with a broad
`pkill -f`. CPU work never waits for the GPU.

Balancing GPU time is your job. Your verification runs come first; give the
two subagents the rest in short jobs (a kernel microbenchmark or `ncu` pass is
minutes, a runtime experiment is tens of minutes at most), split fairly over a
cycle, and write the current plan in `design/MACRO.md`.
`flock -n /tmp/gpu.lock true` tells anyone whether the card is free.

### Engineering practice

Read the supplied vLLM/SGLang experiment notes and working-system lessons
(`references/13-vllm-v1.md`, `14-sglang-runtime.md`).
Extract applicable mechanisms and limits rather than copying an architecture.
Use seven responsibility axes: deployment, semantics, state_memory, scheduling,
host_io, device_compute, topology_transfer. State one falsifiable hypothesis and
the affected invariants, measured before/after quantities and bounds.
Reuse still-valid evidence for unaffected axes; unknown is not a pass.
`scripts/design_review.py` checks this compact record, not a proof for all inputs.

Pick each kernel's language by measurement: Triton for quick iteration, and
CUDA C++ or PTX where Triton cannot reach the target -- warp-level control,
explicit tensor-core `mma` on the card's architecture, persistent decode
kernels, or launch-overhead-bound small batches. Both toolchains work here
(`nvcc` and torch share CUDA 12.8). A compiled extension is built by
`serve.sh` at every cold start, inside the verdict's 900-second readiness
limit; budget its compile time.

Profile real candidate GPU execution with `torch.profiler`, `nsys` or `ncu`.
Microbenchmark **both short requests and extreme-length requests** within the
task envelope, including prefill and long decode tails. If the largest input
and output cannot coexist, test their permitted extreme combinations separately.
Record input and output lengths, memory, timing, correctness observations and
source identity.

`scripts/run_trial.py --through correctness` gives source-bound cold-start,
protocol and replay evidence with the same delivery filter as publication.
Use `profile_candidate.py` and `design_review.py` as described in
`references/18-design-review.md`. Evidence belongs below `/workspace/agentx`
and must identify the source actually tested (`candidate_tree_sha`). Never relabel stale evidence.

### The full test

**Run `./full-test` at every verification, not only at the end.** Stop your
own GPU servers and profilers, then run
`flock /tmp/gpu.lock ./full-test --duration N --wait` early enough for cold
startup, the N-second window and repairs. It freezes the delivery bundle
exactly as publication would, cold-starts that bundle's `serve.sh` inside this
container on its own port, runs the protocol/context probes and an N-second
replay at the bundle's declared concurrency with the evaluator in
`tests/final_verdict/`, probes the same server again after the window,
reports `dollars_per_dollar`, and stops it. It fails a window in which fewer
than one request per minute completed or whose server log records an
unhandled exception or CUDA out-of-memory error. It does not check output
correctness, which stays private to the final verdict.
Reports and logs appear under `/workspace/agentx/full-tests/`. A startup
failure, an incomplete run, or a window that completes almost no requests is
not a pass. Inspect every failure; rerun after repairing material defects.
`run_trial.py --stage full --design-evidence RECEIPT` measures performance and
actual-workload replay fidelity, not output correctness. No profiler, extra
model requests or diagnostic instrumentation belongs in a timed serving window.

### Delivery and budget

Read `DELIVERY_POLICY.json` and `DELIVERY_REPORT.json`: exclusions cover binary
assets, build products, data, archives, evidence, hidden paths, symlinks, size and
depth limits—not just `.venv`. Declare source-relative dependencies in
`runtime_resources.json` (`required` list; `generated` maps output paths to
delivered startup-recipe paths). Excluded resources must be reproducibly created
by `serve.sh` before readiness; declaring a recipe does not execute it.
No dependency on a development cache or temporary file may be assumed.

Use `./clock` to choose experiments that fit. At timebox the harness selects
the last working tree; use a concrete `./promote VERSION` only for an
intentional rollback. Record a concrete blocker when no safe work remains.

Protocol/context checks and lightweight replay establish neither output
correctness nor a score. Only complete
source-bound evidence supports the claim it actually measured.

## Versions

`./checkpoint "why"` freezes the working tree as an immutable version and returns immediately. It does not run evaluation and does not move the final head. Capture/materialization runs asynchronously; checkpointing requires no full benchmark or gate pass.

At timebox the harness freezes and selects the last working tree, so final edits are not lost. A concrete `./promote <version-id> "why"` is an explicit rollback override to an older fallback; `promote latest` is optional bookkeeping. Stateful-serving changes should pass the lightweight replay gate on that exact source SHA; a full replay also satisfies it, but a full window is not required.

Evidence intended to select a version must name its immutable candidate tree SHA. Unbound endpoint diagnostics cannot be relabelled as a newer checkpoint's result. After a checkpoint is frozen, `BOUNDARY_CHECKS.md` records its portability checks and mandatory source-bound CUDA profiling evidence. An unverified checkpoint is retained for recovery, not a verified iteration.

## Budget

This run has a fixed deadline, recorded as `deadline_utc` in `WORKSPACE.json`
and reported live by `./clock`. At the deadline the agent is stopped and the verdict is taken.

## Continuation

Continuing: the same session and the same workspace; nothing has been reset.
Budget remaining {budget_hours} h against this run's deadline of
{deadline_utc}, which has not moved.

### Instructed loop reminder

Run the evolve loop -- plan, request, build, integrate, **verify**, decide --
with both subagents working in the background, each in its own git worktree,
and send each its next assignment when it returns; every hand-off names the
commit it measured. Keep the failure modes in view: correct alone but wrong
under load, a check that cannot fail, shipping what was not measured, numbers
explained away, a guessed offered load, latency lost to prefill between decode
steps, an engine that dies silently, and idle time.

A cycle ends only at a verified tree, cold-started through `serve.sh` (the
only thing the verdict starts):
- correct by
your own output-correctness checks
  against an independent reference of the checkpoint, through the HTTP
  endpoint, at the longest contexts, with prefix reuse, forks and requests
  joining and leaving a concurrent batch, long decodes, continuation past
  end-of-sequence under `ignore_eos` and a natural stop without it, plus the
  protocol probes (`--check-protocol-context`) and the lightweight replay gate;
- runnable and measured by
`flock /tmp/gpu.lock ./full-test --duration N --wait`
  at the concurrency in `concurrency.json`, with every request completing and
  every surprise in the server log and per-request export explained;
  performance decisions rest on repeated, long-enough windows, never the best
  one, and the offered load is chosen by measuring several.
Then `./checkpoint "verified: ..."`; otherwise repair or revert to the last
verified checkpoint. Profile real CUDA execution (torch.profiler, nsys or ncu),
bind evidence to the
immutable candidate SHA
it was measured on, and use design_review.py to update the affected axes.

Every GPU job runs under `flock /tmp/gpu.lock` for exactly its own lifetime,
one agent at a time, and you balance GPU time; CPU work overlaps freely, and
waiting is time for CPU work. In the last hour add nothing new: verify the
final tree with an 1800-second full test and the correctness suite, and use the
whole budget. Timebox selects the
last working tree,
with a concrete `./promote` only for rollback. Unmeasured is not a pass.

Use ./gpu-clock check for read-only diagnosis; low idle clocks are not proof
of a fault. Never recover another job's GPU or recover during measurement.

## Idle restart

The same session, resumed. The previous turn ended without calling a tool that
changed anything. Budget remaining {budget_hours} h against this run's
deadline of {deadline_utc}, which has not moved.

Two facts about this environment, because a turn that ends on either of them
ends the work until the harness relaunches it. Nothing here schedules a future
invocation of the agent: there is no timer, no cron, no wakeup and no
notification channel, and the tools that name one are not available in this
container. And a turn is the only place work happens -- between turns no code
of the agent's runs, though processes it started keep running. What blocks
inside a turn is `wait/`, whose waiters return when their condition holds, and
any command run in the foreground.

### Instructed loop reminder

Run the evolve loop -- plan, request, build, integrate, **verify**, decide --
with both subagents working in the background, each in its own git worktree,
and send each its next assignment when it returns; every hand-off names the
commit it measured. Keep the failure modes in view: correct alone but wrong
under load, a check that cannot fail, shipping what was not measured, numbers
explained away, a guessed offered load, latency lost to prefill between decode
steps, an engine that dies silently, and idle time.

A cycle ends only at a verified tree, cold-started through `serve.sh` (the
only thing the verdict starts):
- correct by
your own output-correctness checks
  against an independent reference of the checkpoint, through the HTTP
  endpoint, at the longest contexts, with prefix reuse, forks and requests
  joining and leaving a concurrent batch, long decodes, continuation past
  end-of-sequence under `ignore_eos` and a natural stop without it, plus the
  protocol probes (`--check-protocol-context`) and the lightweight replay gate;
- runnable and measured by
`flock /tmp/gpu.lock ./full-test --duration N --wait`
  at the concurrency in `concurrency.json`, with every request completing and
  every surprise in the server log and per-request export explained;
  performance decisions rest on repeated, long-enough windows, never the best
  one, and the offered load is chosen by measuring several.
Then `./checkpoint "verified: ..."`; otherwise repair or revert to the last
verified checkpoint. Profile real CUDA execution (torch.profiler, nsys or ncu),
bind evidence to the
immutable candidate SHA
it was measured on, and use design_review.py to update the affected axes.

Every GPU job runs under `flock /tmp/gpu.lock` for exactly its own lifetime,
one agent at a time, and you balance GPU time; CPU work overlaps freely, and
waiting is time for CPU work. In the last hour add nothing new: verify the
final tree with an 1800-second full test and the correctness suite, and use the
whole budget. Timebox selects the
last working tree,
with a concrete `./promote` only for rollback. Unmeasured is not a pass.

Use ./gpu-clock check for read-only diagnosis; low idle clocks are not proof
of a fault. Never recover another job's GPU or recover during measurement.

## Cold restart

The previous session ended and could not be resumed, so this context is new
and nothing precedes this message. What survives is the workspace: its code,
its measurements, its notes, and `WORKSPACE.json`. Budget remaining
{budget_hours} h against this run's deadline of {deadline_utc}, which was
set when the run was created and has not moved.

### Instructed loop reminder

Run the evolve loop -- plan, request, build, integrate, **verify**, decide --
with both subagents working in the background, each in its own git worktree,
and send each its next assignment when it returns; every hand-off names the
commit it measured. Keep the failure modes in view: correct alone but wrong
under load, a check that cannot fail, shipping what was not measured, numbers
explained away, a guessed offered load, latency lost to prefill between decode
steps, an engine that dies silently, and idle time.

A cycle ends only at a verified tree, cold-started through `serve.sh` (the
only thing the verdict starts):
- correct by
your own output-correctness checks
  against an independent reference of the checkpoint, through the HTTP
  endpoint, at the longest contexts, with prefix reuse, forks and requests
  joining and leaving a concurrent batch, long decodes, continuation past
  end-of-sequence under `ignore_eos` and a natural stop without it, plus the
  protocol probes (`--check-protocol-context`) and the lightweight replay gate;
- runnable and measured by
`flock /tmp/gpu.lock ./full-test --duration N --wait`
  at the concurrency in `concurrency.json`, with every request completing and
  every surprise in the server log and per-request export explained;
  performance decisions rest on repeated, long-enough windows, never the best
  one, and the offered load is chosen by measuring several.
Then `./checkpoint "verified: ..."`; otherwise repair or revert to the last
verified checkpoint. Profile real CUDA execution (torch.profiler, nsys or ncu),
bind evidence to the
immutable candidate SHA
it was measured on, and use design_review.py to update the affected axes.

Every GPU job runs under `flock /tmp/gpu.lock` for exactly its own lifetime,
one agent at a time, and you balance GPU time; CPU work overlaps freely, and
waiting is time for CPU work. In the last hour add nothing new: verify the
final tree with an 1800-second full test and the correctness suite, and use the
whole budget. Timebox selects the
last working tree,
with a concrete `./promote` only for rollback. Unmeasured is not a pass.

Use ./gpu-clock check for read-only diagnosis; low idle clocks are not proof
of a fault. Never recover another job's GPU or recover during measurement.

## Make-up run

The provider conversation persisted, and this is a newly identified run whose
workspace was rebuilt from this lineage's published head. `AGENT_TASK.md` is
the specification. Budget remaining {budget_hours} h against this run's
deadline of {deadline_utc}.

### Instructed loop reminder

Run the evolve loop -- plan, request, build, integrate, **verify**, decide --
with both subagents working in the background, each in its own git worktree,
and send each its next assignment when it returns; every hand-off names the
commit it measured. Keep the failure modes in view: correct alone but wrong
under load, a check that cannot fail, shipping what was not measured, numbers
explained away, a guessed offered load, latency lost to prefill between decode
steps, an engine that dies silently, and idle time.

A cycle ends only at a verified tree, cold-started through `serve.sh` (the
only thing the verdict starts):
- correct by
your own output-correctness checks
  against an independent reference of the checkpoint, through the HTTP
  endpoint, at the longest contexts, with prefix reuse, forks and requests
  joining and leaving a concurrent batch, long decodes, continuation past
  end-of-sequence under `ignore_eos` and a natural stop without it, plus the
  protocol probes (`--check-protocol-context`) and the lightweight replay gate;
- runnable and measured by
`flock /tmp/gpu.lock ./full-test --duration N --wait`
  at the concurrency in `concurrency.json`, with every request completing and
  every surprise in the server log and per-request export explained;
  performance decisions rest on repeated, long-enough windows, never the best
  one, and the offered load is chosen by measuring several.
Then `./checkpoint "verified: ..."`; otherwise repair or revert to the last
verified checkpoint. Profile real CUDA execution (torch.profiler, nsys or ncu),
bind evidence to the
immutable candidate SHA
it was measured on, and use design_review.py to update the affected axes.

Every GPU job runs under `flock /tmp/gpu.lock` for exactly its own lifetime,
one agent at a time, and you balance GPU time; CPU work overlaps freely, and
waiting is time for CPU work. In the last hour add nothing new: verify the
final tree with an 1800-second full test and the correctness suite, and use the
whole budget. Timebox selects the
last working tree,
with a concrete `./promote` only for rollback. Unmeasured is not a pass.

Use ./gpu-clock check for read-only diagnosis; low idle clocks are not proof
of a fault. Never recover another job's GPU or recover during measurement.
