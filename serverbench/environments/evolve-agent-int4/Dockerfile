# The agent image for tasks whose checkpoint is compressed-tensors quantized
# (qwen3_8_27b_int4_standard_2xL40: RedHatAI/Qwen3.8-27B-INT4, pack-quantized W4A16).
#
# The shared agent image (serverbench/evolve-agent:cu128) has transformers 5.16.1,
# which refuses to load such a checkpoint without compressed-tensors>=0.15.0:
#     ImportError: compressed-tensors>=0.15.0 is required for compressed-tensors
#     quantization.
# so the starter could not load its own model there. This image adds exactly that
# package, plus loguru (its one dependency the base lacks), pinned by hash and
# installed with --no-deps: torch, transformers and everything else stay the
# base image's. 0.15.0.1 is the newest release that accepts the base's
# torch 2.8 (0.16.0 and later require torch>=2.10). Other tasks keep using
# the base image unchanged.
#
#   docker build -t serverbench/evolve-agent:cu128-int4 serverbench/environments/evolve-agent-int4
FROM serverbench/evolve-agent:cu128
USER root
RUN printf '%s\n' \
      'compressed-tensors==0.15.0.1 --hash=sha256:e1b1f322e82e475715e242bad46925a304ea8e5c98b5055a15b8eb22fb6bfea9' \
      'loguru==0.7.3 --hash=sha256:31a33c10c8e1e10422bfd431aeb5d351c7cf7fa671e3c4df004162264b28220c' > /tmp/int4-requirements.txt \
 && PIP_NO_INDEX=0 pip install --no-cache-dir --no-deps --require-hashes \
      -r /tmp/int4-requirements.txt \
 && rm /tmp/int4-requirements.txt \
 && python3 -c "import compressed_tensors, loguru, torch, transformers; \
      print('int4 deps:', compressed_tensors.__version__, 'torch', torch.__version__, \
            'transformers', transformers.__version__)"
USER agent
