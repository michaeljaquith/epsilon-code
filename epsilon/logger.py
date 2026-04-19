"""
Persistent session logging for the epsilon library.
Writes a structured log file in JSONL format (one JSON object per line).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import EpsilonResult


DEFAULT_LOG_PATH = Path("epsilon_session.log")


class EpsilonLogger:
    """Writes and reads the JSONL session log."""

    def __init__(self, log_path: str | Path = DEFAULT_LOG_PATH):
        self.log_path = Path(log_path)

    def append(self, prompt: str, result: "EpsilonResult", context: str = "", embedding: list | None = None) -> None:
        """Append one log entry to the file.

        context and embedding are optional — callers that do not have an embedding
        (e.g. manual calls from notebooks) simply omit them and the entry will
        not participate in K-NN retrieval but will still contribute to the flat
        fallback pool.
        """
        entry = {
            "timestamp":         datetime.now().isoformat(timespec="seconds"),
            "prompt":            prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "context":           context[:100] if context else "",
            "model":             result.model,
            "epsilon_file":      round(result.epsilon_file, 4),
            "status":            result.status,
            "prompt_tokens":     result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "n_code_tokens":     result.n_code_tokens,
            "epsilon_by_func":   {k: round(v, 4) for k, v in result.epsilon_by_func.items()},
            "peaks": [
                {
                    "token":       te.token,
                    "line":        te.line,
                    "epsilon":     round(te.epsilon, 4),
                    "probability": round(te.probability, 4),
                    "top_alts":    [(t, round(p, 4)) for t, p in te.top_alternatives[:3]],
                }
                for te in result.peak_tokens
            ],
            "flags": result.flags[:5],
        }
        if embedding is not None:
            entry["embedding"] = embedding   # 1536 floats, ~6 KB per entry

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        """Return all log entries as a list of dicts."""
        if not self.log_path.exists():
            return []
        entries = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def get_neighborhood(self, query_embedding: list[float], k: int) -> tuple[list[float], int]:
        """Return (epsilon_values, total_n_code_tokens) for the k most similar runs.

        Similarity is cosine distance in the text-embedding-3-small space.
        Requires numpy. If numpy is unavailable, falls back to the most recent k
        entries (equivalent to the old flat-pool behavior).

        Entries without an 'embedding' field (written by older code or manual
        EpsilonLogger.append calls) are excluded from cosine ranking but the
        method still returns a useful neighborhood when no embedded entries exist
        by falling back to the most recent k entries.
        """
        entries = self.read_all()
        if not entries:
            return [], 0

        try:
            import numpy as np
        except ImportError:
            recent   = entries[-k:]
            epsilons = [e["epsilon_file"] for e in recent if "epsilon_file" in e]
            n_tokens = sum(e.get("n_code_tokens", 0) for e in recent)
            return epsilons, n_tokens

        embedded = [(e, np.array(e["embedding"], dtype=np.float32))
                    for e in entries if e.get("embedding")]

        if not embedded:
            # No embeddings in log yet — fall back to most recent k entries
            recent   = entries[-k:]
            epsilons = [e["epsilon_file"] for e in recent if "epsilon_file" in e]
            n_tokens = sum(e.get("n_code_tokens", 0) for e in recent)
            return epsilons, n_tokens

        # Vectorised cosine similarity
        q      = np.array(query_embedding, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        vecs   = np.stack([v for _, v in embedded])
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        sims   = (vecs / norms) @ q_norm          # shape: (n_embedded,)

        n_take = min(k, len(embedded))
        if n_take == len(embedded):
            top_idx = np.argsort(sims)[::-1]
        else:
            top_idx = np.argpartition(sims, -n_take)[-n_take:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        top_entries = [embedded[i][0] for i in top_idx]
        epsilons    = [e["epsilon_file"] for e in top_entries if "epsilon_file" in e]
        n_tokens    = sum(e.get("n_code_tokens", 0) for e in top_entries)
        return epsilons, n_tokens

    def get_recent_epsilons(self, n: int) -> list[float]:
        """Return the last n file-level ε values from the log, oldest first."""
        entries = self.read_all()
        recent  = entries[-n:] if len(entries) > n else entries
        return [e["epsilon_file"] for e in recent if "epsilon_file" in e]

    def print_summary(self, last_n: int = 20) -> None:
        """Print the last N log entries in a human-readable format."""
        entries = self.read_all()[-last_n:]
        if not entries:
            print(f"No entries found in {self.log_path}")
            return

        print(f"\nSession log: {self.log_path}  ({len(entries)} entries shown)\n")
        print(f"{'Timestamp':22}  {'ε':>7}  {'Status':10}  {'Out':>5}  Prompt")
        print("-" * 90)
        for e in entries:
            ts = e.get("timestamp", "")
            print(
                f"{ts:22}  {e.get('epsilon_file', 0):>7.4f}  "
                f"{e.get('status', ''):10}  {e.get('completion_tokens', 0):>5}  "
                f"{e.get('prompt', '')[:50]}"
            )
