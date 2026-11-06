"""Model backends. A backend turns (messages, tools) into the raw text of one assistant turn."""

from __future__ import annotations

import copy
import hashlib
import os
import time
from pathlib import Path

PREFIX_END = "<|im_start|>user"


def evict_lru(folder: Path, max_bytes: int, keep: Path | None = None) -> int:
    """Delete the least recently used cache files until the folder is under max_bytes; returns files removed."""
    files = sorted(folder.glob("*.safetensors"), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    removed = 0
    for f in files:
        if total <= max_bytes:
            break
        if keep is not None and f == keep:
            continue
        total -= f.stat().st_size
        f.unlink()
        removed += 1
    return removed


class PromptCache:
    """Reuses model state across calls whose prompts extend each other (the steps of an agent).

    After a prompt is prefilled up to its last token, a copy of the cache is kept with those
    tokens. A later prompt that starts with exactly the same tokens copies it and prefills only
    the new ones. Gated DeltaNet state cannot be trimmed back, so reuse needs an exact token
    prefix; anything else falls back to a full prefill, which keeps the output correct.
    With prefix_dir, the cache of the system message plus tool schemas (everything before the
    first user message) is stored on disk under the hash of its tokens and loaded from there.
    The folder is capped (max_disk_gb, least recently used files go first): a 4B prefix is about
    120 MB, and with a different system prompt per conversation an uncapped folder filled a disk.
    """

    def __init__(self, model, prefix_dir: str | None = None, step: int = 2048, max_disk_gb: float = 4.0):
        self.model = model
        self.dir = Path(prefix_dir).expanduser() if prefix_dir else None
        self.max_bytes = int(max_disk_gb * 1e9)
        self.step = step
        self.tokens: list[int] | None = None
        self.cache = None
        self.stats = {"memory_hits": 0, "disk_hits": 0, "disk_writes": 0, "disk_evictions": 0, "misses": 0}

    def _prefill(self, cache, toks):
        import mlx.core as mx
        for i in range(0, len(toks), self.step):
            self.model(mx.array([toks[i:i + self.step]]), cache=cache)
            mx.eval([c.state for c in cache])

    def _prefix(self, prefix):
        from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
        key = hashlib.sha256(" ".join(map(str, prefix)).encode()).hexdigest()[:32]
        f = self.dir / f"{key}.safetensors"
        if f.exists():
            self.stats["disk_hits"] += 1
            os.utime(f)  # a hit makes it recently used
            return load_prompt_cache(str(f))
        cache = make_prompt_cache(self.model)
        self._prefill(cache, prefix)
        self.dir.mkdir(parents=True, exist_ok=True)
        save_prompt_cache(str(f), cache, {"tokens": str(len(prefix))})
        self.stats["disk_writes"] += 1
        self.stats["disk_evictions"] += evict_lru(self.dir, self.max_bytes, keep=f)
        return cache

    def prepare(self, prompt: list[int], prefix_len: int | None) -> tuple[object, int]:
        """A cache holding prompt[:-1]; returns it and how many tokens came from reuse."""
        from mlx_lm.models.cache import make_prompt_cache
        body = prompt[:-1]
        if self.tokens is not None and len(self.tokens) <= len(body) and body[:len(self.tokens)] == self.tokens:
            cache, start = copy.deepcopy(self.cache), len(self.tokens)
            self.stats["memory_hits"] += 1
        elif self.dir is not None and prefix_len and prefix_len <= len(body):
            cache, start = self._prefix(body[:prefix_len]), prefix_len
        else:
            cache, start = make_prompt_cache(self.model), 0
            self.stats["misses"] += 1
        self._prefill(cache, body[start:])
        self.tokens, self.cache = list(body), copy.deepcopy(cache)
        return cache, start


class MLXBackend:
    """In-process mlx-lm, chat template of the model, thinking off, greedy by default."""

    def __init__(self, model: str, temp: float = 0.0, max_tokens: int = 1536, seed: int = 0,
                 top_p: float = 1.0, top_k: int = 20, presence_penalty: float = 0.0, thinking: bool = False,
                 cache_reuse: bool = False, prefix_dir: str | None = None, adapter: str | None = None):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        self.mx = mx
        self.model, self.tok = load(model, adapter_path=adapter)
        self.name = model if adapter is None else f"{model}+{adapter}"
        self.temp = temp
        self.max_tokens = max_tokens
        self.seed = seed
        self.thinking = thinking
        self.sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k if temp > 0 else 0)
        self.processors = make_logits_processors(presence_penalty=presence_penalty) if presence_penalty else None
        self.pcache = PromptCache(self.model, prefix_dir) if (cache_reuse or prefix_dir) else None

    def clone(self, **kw) -> "MLXBackend":
        """Another decoding setup on the same loaded model. Sampling parameters not given take the values Qwen
        recommends for sampling (top_p 0.95, top_k 20, no presence penalty), not the parent's."""
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        twin = object.__new__(MLXBackend)
        twin.__dict__.update(self.__dict__)
        twin.temp = kw.get("temp", self.temp)
        twin.thinking = kw.get("thinking", self.thinking)
        twin.max_tokens = kw.get("max_tokens", self.max_tokens)
        twin.seed = kw.get("seed", self.seed)
        twin.sampler = make_sampler(temp=twin.temp, top_p=kw.get("top_p", 0.95 if twin.temp > 0 else 1.0),
                                    top_k=kw.get("top_k", 20) if twin.temp > 0 else 0)
        pp = kw.get("presence_penalty", 0.0)
        twin.processors = make_logits_processors(presence_penalty=pp) if pp else None
        if not kw.get("cache", True):
            twin.pcache = None  # e.g. a writer whose unrelated prompts would only evict the agent's snapshot
        return twin

    def parse_tool_call(self, body: str, tools: list[dict]) -> dict:
        return self.tok.tool_parser(body, tools)

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        from mlx_lm import stream_generate

        text_prompt = self.tok.apply_chat_template(
            messages, tools=tools or None, add_generation_prompt=True, tokenize=False, enable_thinking=self.thinking
        )
        if self.thinking:
            # The generation prompt already opened <think>; give the parser the whole block back.
            text_prompt_tail = "<think>\n"
        else:
            text_prompt_tail = ""
        prompt = self.tok.encode(text_prompt, add_special_tokens=False)
        if self.temp > 0:
            self.mx.random.seed(self.seed)
        t = time.perf_counter()
        extra, reused = {}, 0
        if self.pcache is not None:
            prefix_len = None
            if PREFIX_END in text_prompt:
                pre = self.tok.encode(text_prompt[:text_prompt.index(PREFIX_END)], add_special_tokens=False)
                prefix_len = len(pre) if prompt[:len(pre)] == pre else None
            cache, reused = self.pcache.prepare(prompt, prefix_len)
            extra = {"prompt_cache": cache}
            feed = prompt[-1:]
        else:
            feed = prompt
        text, n_gen = [], 0
        for r in stream_generate(self.model, self.tok, feed, max_tokens=self.max_tokens, sampler=self.sampler,
                                 logits_processors=self.processors, **extra):
            text.append(r.text)
            n_gen = r.generation_tokens
        return {
            "text": text_prompt_tail + "".join(text),
            "prompt_tokens": len(prompt),
            "reused_tokens": reused,
            "gen_tokens": n_gen,
            "hit_max_tokens": n_gen >= self.max_tokens,
            "seconds": time.perf_counter() - t,
        }
