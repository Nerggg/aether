# Aether (rewrite)

A stateful LLM game master for locally hosted tabletop RPGs, powered by a small local model through Ollama.

> **The rewrite is still in progress.**
> Please refer to [the main branch](https://github.com/Nerggg/aether/tree/main) for a working application.

## Why a rewrite?

The original version on `main` works, but its structure made it hard to extend. The game state was split across SQLite and Markdown files, the narration loop had no long-term memory, and parts of the design were never finished. The rewrite starts from a clean foundation with a single SQLite database per campaign, a typed and tested game-rules layer, and a turn loop built for a 3B-parameter model.

## Status

| Area | State |
| --- | --- |
| Settings and configuration | Done |
| SQLite schema and auto-save | Done |
| Game rules (stats, dice, tag parsing, story beats) | Done |
| LLM client and structured (Pydantic) JSON calls | Done |
| Campaign creation | Done |
| Character creation | Done |
| Game loop (opening narration, options, action judging, dice, auto-save) | Working, being tuned |
| Rolling summary (memory compaction) | Planned |
| Transcript retrieval (RAG) | Planned |
| Main menu and resume polish | Planned |
| Full documentation | Planned |

## Tests

The deterministic parts of the engine (database, stats, tag parsing, story logic, option cleanup, prompt building and the creation flows) are covered by a pytest suite that uses a fake LLM backend, so it runs without a model.

```
pip install -r requirements-dev.txt
pytest
```

## Documentation

Full setup and usage instructions will be written once the rewrite is complete. Until then, the README on `main` describes the original application.
