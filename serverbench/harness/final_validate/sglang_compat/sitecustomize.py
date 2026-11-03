"""Narrow compatibility fix loaded by Python in the pinned SGLang container.

SGLang v0.5.19's ``Gemma4UnifiedForConditionalGeneration.__init__`` computes
``text_tie`` and uses it to construct the output head, but never stores the
``lm_head_is_tied`` attribute inherited ``forward`` reads. The first Gemma 4
request therefore crashes in both eager and CUDA-graph execution.

This restores the missing assignment without changing weights, kernels,
scheduling, or request semantics. It is intentionally conditional on the one
class and on the attribute being absent, so other SGLang models are untouched.
"""

try:
    from sglang.srt.models.gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
    )

    _original_init = Gemma4UnifiedForConditionalGeneration.__init__

    def _init_with_tied_head_state(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        if not hasattr(self, "lm_head_is_tied"):
            text_config = self.config.text_config
            self.lm_head_is_tied = bool(
                getattr(text_config, "tie_word_embeddings", True))

    Gemma4UnifiedForConditionalGeneration.__init__ = _init_with_tied_head_state
except (ImportError, AttributeError):
    # The pinned image carries the module. Keeping sitecustomize importable on
    # other Python environments makes inspection and repository tests cheap;
    # a missing SGLang module is still reported by the server command itself.
    pass


# The pinned OpenAI chat adapter exposes native IDs only for non-streaming
# responses. Export them on streaming requests too, without touching sampling
# or model computation. Re-encoding decoded text is not invertible: e.g.
# ["Test", "DataLoader"] can re-encode as ["TestData", "Loader"].
try:
    import json

    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    _original_convert = OpenAIServingChat._convert_to_internal_request
    _original_stream = OpenAIServingChat._generate_stream_content

    def _convert_with_stream_ids(self, request, *args, **kwargs):
        wants_ids = bool(request.stream and request.return_token_ids)
        if wants_ids:
            request.return_token_ids = False
        try:
            return _original_convert(self, request, *args, **kwargs)
        finally:
            if wants_ids:
                request.return_token_ids = True

    async def _stream_with_native_ids(
            self, *, content, index, request, stream_offsets, **kwargs):
        wants_ids = bool(request.return_token_ids)
        ids = []
        if wants_ids:
            output_ids = list(content["output_ids"])
            key = ("native_token_ids", index)
            if self.tokenizer_manager.server_args.incremental_streaming_output:
                ids = output_ids
            else:
                ids = output_ids[stream_offsets.get(key, 0):]
                stream_offsets[key] = len(output_ids)
        async for chunk in _original_stream(
                self, content=content, index=index, request=request,
                stream_offsets=stream_offsets, **kwargs):
            if ids:
                event = json.loads(chunk.removeprefix("data: "))
                event["choices"][0]["delta"]["token_ids"] = ids
                chunk = "data: " + json.dumps(event) + "\n\n"
                ids = []
            yield chunk
        if ids:
            # A buffered UTF-8 fragment or a parser can emit no text yet.
            yield "data: " + json.dumps({
                "id": content["meta_info"]["id"],
                "object": "chat.completion.chunk", "model": request.model,
                "choices": [{"index": index, "delta": {"token_ids": ids},
                             "finish_reason": None}],
            }) + "\n\n"

    OpenAIServingChat._convert_to_internal_request = _convert_with_stream_ids
    OpenAIServingChat._generate_stream_content = _stream_with_native_ids
except (ImportError, AttributeError):
    pass
