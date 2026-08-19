"""Check the LLM endpoint before touching the pipeline.

    python scripts/check_llm.py            # list models, then test the configured one
    python scripts/check_llm.py --list     # list models only

Three things can be wrong, and this tells you which:
  1. .env missing or empty        -> ValueError from ChatClient
  2. URL / key wrong              -> connection or 401 on --list
  3. model name wrong             -> --list succeeds, the completion 404s
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.config import load_config  # noqa: E402
from unitn_rag.llm import ChatClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--list", action="store_true", help="list models and stop")
    ap.add_argument("--model", help="override config.yaml's llm.model for this test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.llm.model = args.model

    print(f"base_url : {cfg.llm.base_url or '(not set)'}")
    print(f"api_key  : {'set, ' + str(len(cfg.llm.api_key)) + ' chars' if cfg.llm.api_key else '(not set)'}")
    print(f"model    : {cfg.llm.model}\n")

    try:
        client = ChatClient(cfg.llm, max_retries=1)
    except ValueError as e:
        print(f"FAIL  {e}")
        return 1

    models = client.list_models()
    print(f"models visible at this endpoint ({len(models)}):")
    for m in models:
        print(f"  {'* ' if m == cfg.llm.model else '  '}{m}")

    if args.list:
        return 0

    if models and not any(m == cfg.llm.model for m in models) and "could not list" not in models[0]:
        print(f"\nWARN  '{cfg.llm.model}' is not in that list. Set llm.model in config.yaml to one of them.")

    print("\nsending a test completion...")
    t0 = time.time()
    try:
        answer = client.complete(
            [
                {"role": "system", "content": "Answer in one short sentence."},
                {"role": "user", "content": "In quale citta si trova l'Universita di Trento?"},
            ],
            max_tokens=60,
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {e}")
        return 1

    dt = time.time() - t0
    print(f"OK    {dt:.1f}s\n{answer}\n")

    if not answer:
        print("WARN  empty response - the endpoint replied but the model returned nothing.")
        return 1
    if dt > 20:
        print("WARN  slow. At this latency a 50-question eval run takes "
              f"~{dt * 50 / 60:.0f} minutes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
