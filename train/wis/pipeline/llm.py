from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"

_ENV_KEYS = (
    "LLM_PROVIDER",
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "DEEPSEEK_API_KEY",
)


def _load_dotenv():
    """从仓库根目录 .env 读取缺失的配置，避免把密钥写进命令行或提交进仓库。"""
    dotenv = Path(__file__).resolve().parents[3] / ".env"
    if not dotenv.exists():
        return
    for raw in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in _ENV_KEYS and not os.environ.get(key):
            os.environ[key] = value


_load_dotenv()


def _is_local(url):
    return any(token in url for token in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"))


class LLMClient:
    """统一 chat 客户端，支持两种后端：

    1) Ollama 原生接口（本地模型，默认）—— 走 /api/chat。
       对 qwen3 这类带思考的模型，用 think=False 关掉思维链，实测比 OpenAI 兼容
       端点快约 35 倍（11.3s → 0.3s），这是批量造数据的关键。
       环境变量：OLLAMA_HOST（默认 http://localhost:11434）、OLLAMA_MODEL（默认 qwen3:8b）
    2) OpenAI 兼容接口（远程/云端）—— 走 /chat/completions。
       环境变量：OPENAI_BASE_URL、OPENAI_API_KEY、OPENAI_MODEL

    选择顺序：显式 provider > OLLAMA_HOST > 本地 base_url > OPENAI_API_KEY > 本地 Ollama。
    """

    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        timeout=300,
        retries=6,
        provider=None,
        host=None,
        think=False,
    ):
        env_host = host or os.environ.get("OLLAMA_HOST")
        env_base = base_url or os.environ.get("OPENAI_BASE_URL")
        env_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        self.provider = provider or os.environ.get("LLM_PROVIDER")
        if self.provider in (None, "", "auto"):
            if env_host:
                self.provider = "ollama"
            elif env_base and _is_local(env_base):
                self.provider = "ollama"
            elif env_key:
                self.provider = "openai"
            else:
                self.provider = "ollama"

        self.think = bool(think)
        self.timeout = timeout
        self.retries = retries
        self._think_supported = None

        if self.provider == "ollama":
            if env_host:
                resolved = env_host
            elif env_base and _is_local(env_base):
                # 只有本地 base_url 才能当 Ollama 用；远端（如 DeepSeek）会 401
                resolved = env_base.rstrip("/")
                if resolved.endswith("/v1"):
                    resolved = resolved[:-3]
            else:
                resolved = DEFAULT_OLLAMA_HOST
            self.host = resolved.rstrip("/")
            self.base_url = self.host
            self.api_key = env_key or "ollama"
            self.model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        else:
            self.base_url = (env_base or "https://api.openai.com/v1").rstrip("/")
            self.api_key = env_key
            self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # ---- 可用性 ----------------------------------------------------------
    def available(self):
        if self.provider == "ollama":
            return self._ollama_alive()
        return bool(self.api_key)

    def describe(self):
        return f"{self.provider}:{self.model} @ {self.host if self.provider == 'ollama' else self.base_url}"

    def _ollama_alive(self):
        try:
            request = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
            names = {m.get("name") for m in body.get("models", [])}
            return self.model in names
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            return False

    def _model_capabilities(self):
        try:
            payload = json.dumps({"model": self.model}).encode("utf-8")
            request = urllib.request.Request(
                f"{self.host}/api/show",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                body = json.loads(response.read().decode("utf-8"))
            return set(body.get("capabilities") or [])
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            return set()

    # ---- 调用 ------------------------------------------------------------
    def chat(self, system, user, temperature=0.7, max_tokens=1024):
        if self.provider == "ollama":
            return self._chat_ollama(system, user, temperature, max_tokens)
        return self._chat_openai(system, user, temperature, max_tokens)

    def chat_json(self, system, user, temperature=0.4, max_tokens=1024):
        text = self.chat(system, user, temperature=temperature, max_tokens=max_tokens)
        return _extract_json(text)

    def _chat_ollama(self, system, user, temperature, max_tokens):
        if self._think_supported is None:
            caps = self._model_capabilities()
            self._think_supported = "thinking" in caps
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if self._think_supported:
            payload["think"] = self.think
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body.get("message", {}).get("content", "")
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Ollama 调用失败（{self.host} / {self.model}）: {last_error}")

    def _chat_openai(self, system, user, temperature, max_tokens):
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY 未设置，OpenAI 后端不可用")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if "deepseek" in self.base_url or "deepseek" in self.model:
            # DeepSeek 推理模型默认先产思考链，常把 max_tokens 吃光导致 content 为空
            payload["thinking"] = {"type": "enabled" if self.think else "disabled"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        last_error = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                choice = body["choices"][0]
                content = choice["message"].get("content") or ""
                if not content.strip():
                    raise RuntimeError(
                        f"空响应（finish_reason={choice.get('finish_reason')}, max_tokens={max_tokens}）"
                    )
                return content
            except urllib.error.HTTPError as exc:
                # 4xx（除限流）重试无意义，直接失败，避免批量任务在坏请求上空转
                if exc.code not in (429, 500, 502, 503, 504):
                    detail = exc.read().decode("utf-8", "ignore")[:200]
                    raise RuntimeError(f"LLM 调用失败 HTTP {exc.code}: {detail}") from exc
                last_error = exc
                # 429 是限流：优先服从服务端 Retry-After，否则指数退避（上限 30s）+ 抖动
                header = (exc.headers.get("Retry-After") if exc.headers else None) or ""
                delay = float(header) if header.strip().isdigit() else min(30.0, 2.0 * (2**attempt))
                time.sleep(delay + random.uniform(0, 1.5))
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(min(30.0, 2.0 * (2**attempt)) + random.uniform(0, 1.5))
        raise RuntimeError(f"LLM 调用失败: {last_error}")


def _extract_json(text):
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None