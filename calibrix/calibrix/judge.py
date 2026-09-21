# Calibrix LLM-judge scorer: remote (HF hosted / OpenAI-compat) + offline.

# Gap patched: Calibri needs heavyweight local reward servers (HPSv3,
# Q-Align); Heretic relies on keyword lists. A judge scorer closes the gap
# for any modality and any deployment: it grades outputs 0..1 using a model
# as the judge. Two backends:
#   RemoteLLMJudge    - huggingface_hub InferenceClient (free tier) or any
#                       OpenAI-compatible endpoint (Ollama on your machine).
#   OfflineJudge      - zero-network heuristic judge (overlap + structure
#                       checks) so search works fully offline.

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, List, Optional, Sequence

from .scorers import Prompt, Scorer, ScorerContext, Score


JUDGE_SYSTEM = (
    "You are a strict grader. Given a TASK and a RESPONSE, output only a "
    "number between 0.0 and 1.0 for how well the response fulfills the task. "
    "No other text."
)


def _grade_from_text(text: str) -> float:
    m = re.search(r"([01](?:\.\d+)?)", text)
    return float(m.group(1)) if m else 0.0


def _post_json(url: str, payload: dict, api_key: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class RemoteLLMJudge(Scorer):
    """Judge via a hosted or local OpenAI-compatible chat model.

    optimization: "maximize" (quality 0..1).

    Works with:
      * HF Inference Client models via router.huggingface.co (free tier)
      * Ollama:  base_url="http://127.0.0.1:11434/v1", model="llama3.2"
      * llama.cpp / LM Studio / vLLM — anything speaking /chat/completions
    """

    optimization = "maximize"
    reproducible = False  # external service

    def __init__(
        self,
        prompts: Sequence[Prompt],
        base_url: str = "https://router.huggingface.co/v1",
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        api_key: Optional[str] = None,
        criteria: str = "helpfulness and correctness",
    ) -> None:
        self.prompts = list(prompts)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.criteria = criteria

    def _grade_one(self, task: str, response: str) -> float:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"TASK: {task}\n\nRESPONSE: {response}\n\nSCORE:",
                },
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        }
        try:
            data = _post_json(
                self.base_url + "/chat/completions", payload, self.api_key or "hf_"
            )
            return _grade_from_text(data["choices"][0]["message"]["content"])
        except Exception:
            return 0.0  # network errors must not crash a search run

    def get_score(self, ctx: ScorerContext) -> Score:
        responses = ctx.get_responses(self.prompts)
        grades = [
            self._grade_one(p.user, r) for p, r in zip(self.prompts, responses)
        ]
        mean = sum(grades) / max(len(grades), 1)
        return Score(value=mean, display=f"judge {mean:.3f}")


class OfflineJudge(Scorer):
    """Zero-network heuristic judge: keyword coverage + structure heuristics.

    Not as sharp as an LLM judge, but deterministic, free, offline, and
    fast enough to run inside every search trial. Good as a dense shaping
    reward alongside sparse human-ish judges.
    """

    optimization = "maximize"

    def __init__(self, prompts: Sequence[Prompt], min_words: int = 8) -> None:
        self.prompts = list(prompts)
        self.min_words = int(min_words)

    @staticmethod
    def _key_terms(text: str) -> List[str]:
        stop = {
            "the", "a", "an", "of", "to", "in", "and", "or", "is", "are",
            "for", "on", "with", "how", "what", "why", "do", "i", "you",
        }
        return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stop][:6]

    def _grade_one(self, task: str, response: str) -> float:
        if not response.strip():
            return 0.0
        terms = self._key_terms(task)
        hit = sum(1 for t in terms if t in response.lower())
        coverage = hit / max(len(terms), 1)
        words = len(response.split())
        length_score = min(words / self.min_words, 1.0)
        structure = 0.1 if any(s in response for s in (".", "\n", "-")) else 0.0
        return min(0.6 * coverage + 0.3 * length_score + structure, 1.0)

    def get_score(self, ctx: ScorerContext) -> Score:
        responses = ctx.get_responses(self.prompts)
        grades = [
            self._grade_one(p.user, r) for p, r in zip(self.prompts, responses)
        ]
        mean = sum(grades) / max(len(grades), 1)
        return Score(value=mean, display=f"offline {mean:.3f}")
