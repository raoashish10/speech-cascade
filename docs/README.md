# Where the docs went

This directory used to hold 13 investigation/reference docs directly in
git. They've been moved to S3 for archival (source material for a
possible future writeup) instead of living in this repo — see
`scripts/archive_docs_to_s3.sh` for the exact upload command and
destination. `git log --diff-filter=D -- docs/` on this branch shows the
last commit each file existed in, if you need the actual content back
(they're not gone, just not tracked going forward).

| File | What it covered |
|---|---|
| `kokoro-tts-capacity-fix.md` | The overnight session's headline finding — `kokoro_tts`'s real 24.3% `voice_pipeline` failure rate, root-caused to ONNX Runtime CUDA arena defaults, and the fix. |
| `kokoro-tts-vram-headroom.md` | Investigation into reclaiming VRAM via `nemotron_llm`'s KV cache fraction for a 4th `kokoro_tts` instance — ruled out, findings only. |
| `kv-cache-investigation.md` | Why FP8 KV cache quantization isn't the right fix for `nemotron_llm`'s decode latency — findings only, not implemented. |
| `monitoring.md` | Grafana/Prometheus setup, alert rules, and why `qwen_llm`'s queue/compute metrics are excluded from panels. |
| `nemotron-batch-size-scaling.md` | Measuring `nemotron_llm`'s `max_batch_size` ceiling (8→16, with 24/32 tested and rejected). |
| `nemotron-response-quality.md` | The forced-empty-`<think>` fix that cut think-leakage 46%→13% on the old Nemotron checkpoint. |
| `nemotron-token-cap-investigation.md` | The stop-sequence fix that cut the 96-token cap-hit rate 97%→7%. |
| `nvfp4-candidate-investigation.md` | Why full NVFP4 was chosen over MLP-only for the original Nemotron quantization. |
| `nvfp4-classic-backend-collapse.md` | **Flag this one specifically if you're picking work back up**: confirmed that TensorRT-LLM's classic backend causes near-total generation collapse (0/50 MATH500) on the NVFP4 Nemotron checkpoint — a backend defect, not a quantization one. Never re-verified against the current `qwen_llm` checkpoint, which likely runs the same backend. Possibly still a live, unresolved production risk, not just history. |
| `qwen-llm-migration.md` | The `nemotron_llm` → `qwen_llm` migration: DCGM install bug, sequential-loading fix for a host-RAM OOM, first load test results. |
| `qwen-nvfp4-serving-backend-comparison.md` | Qwen3-8B-NVFP4 across TensorRT-LLM/vLLM/SGLang, including an unresolved checkpoint metadata inconsistency (`config.json` claims 4-bit activations, contradicting the weight-only model card). |
| `tts-replacement-investigation.md` | The full TTS saga: Magpie rejected (never actually deployed), Kokoro went dead, Chatterbox-Turbo chosen over F5-TTS/XTTS-v2/IndexTTS-2.5. |
| `voice-pipeline-queueing.md` | `voice_pipeline` queueing at concurrency=8 — traced to instance-count vs. `kokoro_tts` capacity tradeoffs, plus a real `load_test.py` metrics bug found along the way. |

Other docs in this repo (`README.md`, `deploy/REBUILD.md`,
`deploy/ansible/README.md`) stay in git — this move is specifically for
the narrative investigation writeups, not live operational reference.
