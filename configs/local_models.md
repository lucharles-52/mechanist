# Local models for Mechanist (one B200 each, vLLM)

Serving flags come from the annotation panel (`Projects/scripts/study_runtime.py`, `LLM_PANEL_DEFAULT`), which
validated them on this cluster. Context is longer here because Mechanist's synthesis prompts reach ~75k tokens.

| Model | Config | MAXLEN | PARSER | QUANT | VLLM_EXTRA | MAXSEQS |
|---|---|---|---|---|---|---|
| openai/gpt-oss-120b | p033_pg_local_gptoss.yaml | 131072 | openai_gptoss | | | 16 |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 | p033_pg_local_nemotron.yaml | 131072 | nemotron_v3 | | --trust-remote-code | 16 |
| Qwen/Qwen3.8-27B-FP8 | p033_pg_local_qwen27b.yaml | 131072 | qwen3 | | --language-model-only | 16 |
| nvidia/Qwen3.8-Flash-Next-NVFP4 | p033_pg_local_qwenflash.yaml | 131072 | qwen3 | modelopt | --trust-remote-code --language-model-only --kv-cache-memory-bytes 21474836480 | 16 |
| nvidia/MiniMax-M2.7-NVFP4 | p033_pg_local_minimax.yaml | 98304 | minimax_m2 | | --trust-remote-code | 4 |

MiniMax's 140 GB of weights leave about 30 GB of the B200 for its KV cache, so it gets a shorter context and fewer
concurrent sequences (threads queue rather than fail). Reasoning is on for every model (thinking enabled; gpt-oss
effort medium; MiniMax always reasons). All share `max_tokens: 16000`.

Launch (one run; `SEED` fixes vLLM sampling so repeats differ only by seed):

```bash
MODES=multi TAG=L_<model>_s0 SEED=0 MODEL=<model> PARSER=<parser> MAXLEN=<len> QUANT=<quant> \
  VLLM_EXTRA="<flags>" MAXSEQS=<n> CONFIG=configs/<config> sbatch --time=12:00:00 --export=ALL slurm/run_p033.sbatch
```
