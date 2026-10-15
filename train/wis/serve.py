"""WIS 本地网页客户端后端：只用标准库 http.server 提供 REST + SSE。

复用 chat.Session / stream_generate / parse_labeled_output 与 collator 的
prompt_ids / render_completion，不重复实现模型加载与 CFG 逻辑。

为什么这样设计：
- 模型加载约 60 秒且推理不是线程安全的：加载放后台线程，服务先起来，
  前端轮询 /api/state；生成与重载用同一把全局锁串行化，避免 8GB 显存 OOM。
- SSE 逐 token flush，前端才看得到流式；客户端断开时回调抛 ClientGone 中止生成，
  避免服务端持续写入报错刷屏。
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .chat import Session, parse_labeled_output, stream_generate
from .collator import prompt_ids, render_completion
from .config import ROOT
from .data import read_jsonl

INDEX_FILE = Path(__file__).resolve().parent / "web" / "index.html"
EVAL_FILE = "train/wis/data/real_v2_smoke/wis_sft_eval.jsonl"  # 缺省验证集（本地冒烟）


def _eval_file(state):
    """验证集路径跟随当前配置的 eval_file，缺省回退到本地冒烟验证集。"""
    cfg = state.session.cfg if state.session is not None else None
    return (cfg or {}).get("eval_file") or EVAL_FILE

# Session.__init__ 会读取这些属性；后台线程构建 Session 时要先凑齐一个等价 args。
_ARGS_ATTRS = ("stage", "adapter", "config", "w", "temperature", "top_p",
               "max_new_tokens", "max_prompt_length")
_PARAM_KEYS = ("w", "temperature", "top_p", "max_new_tokens")


# ---------------------------------------------------------------- 设备/显存
def _vram_text():
    """显存占用文案；取不到（CPU/无 torch）时返回 None，前端据此显示占位。"""
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            return f"{allocated:.0f}MiB / {reserved:.0f}MiB"
        return "CPU"
    except Exception:
        return None


# ---------------------------------------------------------------- 全局状态
class AppState:
    def __init__(self, args):
        self.args = args          # 与 Session 共用同一个命名空间，reload 才能带出新 stage
        self.gen_lock = threading.Lock()
        self.session = None
        self.ready = False
        self.loading = True
        self.error = None

    def snapshot(self):
        session = self.session
        model = None
        device = None
        vram = None
        adapter = self.args.adapter
        if session is not None:
            model = session.cfg.get("model_name_or_path") if session.cfg else None
            device = str(session.device) if session.device is not None else None
            vram = _vram_text()
            adapter = str(session.adapter_path) if session.adapter_path else adapter
        return {
            "ready": self.ready,
            "loading": self.loading,
            "error": self.error,
            "stage": self.args.stage,
            "adapter": adapter,
            "model": model,
            "device": device,
            "vram": vram,
            "w": self.args.w,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_new_tokens": self.args.max_new_tokens,
        }


def _load_worker(state):
    """后台构建 Session；SystemExit 也要兜住，否则线程静默死掉、前端永远转圈。"""
    try:
        session = Session(state.args)
        session.load()
        with state.gen_lock:
            state.session = session
            state.ready = True
            state.error = None
    except BaseException as exc:  # noqa: BLE001 - 需要捕获 SystemExit 等非 Exception
        state.ready = False
        state.error = f"模型加载失败：{type(exc).__name__}: {exc}"
        print(f"[serve] {state.error}", flush=True)
    finally:
        state.loading = False
    if state.ready:
        print("[serve] 模型加载完成，界面已就绪。", flush=True)


def _reload_worker(state, stage, adapter):
    """切换 stage 触发重载（约 60s）：持锁期间生成会排队，符合串行化要求。"""
    try:
        with state.gen_lock:
            if state.session is None:
                session = Session(state.args)
                session.load()
                state.session = session
            else:
                state.session.reload(stage, adapter)
            state.ready = True
            state.error = None
    except BaseException as exc:  # noqa: BLE001
        state.ready = False
        state.error = f"模型切换失败：{type(exc).__name__}: {exc}"
        print(f"[serve] {state.error}", flush=True)
    finally:
        state.loading = False
    if state.ready:
        print(f"[serve] 已切换到 {state.args.stage} adapter。", flush=True)


# ---------------------------------------------------------------- SSE 写出
class ClientGone(Exception):
    """客户端断开：用它从回调里抛出，冒泡穿过 stream_generate 中止生成。"""


class _SSE:
    def __init__(self, wfile):
        self._wfile = wfile

    def send(self, event, data):
        frame = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        try:
            self._wfile.write(frame.encode("utf-8"))
            self._wfile.flush()
        except OSError as exc:  # BrokenPipeError / ConnectionAborted 等都是 OSError
            raise ClientGone() from exc


# ---------------------------------------------------------------- 请求处理
def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # 无 Content-Length 的 SSE 靠连接关闭收尾
        server_version = "WISServe/1.0"

        def log_message(self, fmt, *args):  # 轮询/静态请求太吵，静默
            return

        # -------- 通用写出
        def _send_json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                pass

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        # -------- GET
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path in ("/", "/index.html"):
                    self._serve_index()
                elif parsed.path == "/api/state":
                    self._send_json(state.snapshot())
                elif parsed.path == "/api/sample":
                    self._serve_sample(parse_qs(parsed.query))
                else:
                    self._send_json({"error": "未知路径"}, 404)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            except Exception as exc:  # noqa: BLE001
                self._safe_error(f"服务端异常：{type(exc).__name__}: {exc}")

        def do_POST(self):
            parsed = urlparse(self.path)
            try:
                body = self._read_json()
            except (ValueError, UnicodeDecodeError) as exc:
                self._send_json({"error": f"请求体不是合法 JSON：{exc}"}, 400)
                return
            try:
                if parsed.path == "/api/config":
                    self._handle_config(body)
                elif parsed.path == "/api/generate":
                    self._handle_generate(body)
                else:
                    self._send_json({"error": "未知路径"}, 404)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            except Exception as exc:  # noqa: BLE001
                self._safe_error(f"服务端异常：{type(exc).__name__}: {exc}")

        def _safe_error(self, message):
            try:
                self._send_json({"error": message}, 500)
            except Exception:  # noqa: BLE001
                pass

        # -------- 静态首页
        def _serve_index(self):
            try:
                html = INDEX_FILE.read_text(encoding="utf-8")
            except OSError as exc:
                self._send_json({"error": f"前端文件缺失：{exc}"}, 500)
                return
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                pass

        # -------- 验证集样本
        def _serve_sample(self, query):
            rel = _eval_file(state)
            try:
                rows = read_jsonl(ROOT / rel)
            except OSError as exc:
                self._send_json({"error": f"验证集读取失败（{rel}）：{exc}"}, 500)
                return
            if not rows:
                self._send_json({"error": "验证集为空"}, 404)
                return
            raw_n = (query.get("n") or [None])[0]
            if raw_n:
                try:
                    index = int(raw_n)
                except ValueError:
                    self._send_json({"error": f"非法样本序号：{raw_n!r}（应为整数）"}, 400)
                    return
                if not 1 <= index <= len(rows):
                    self._send_json({"error": f"样本序号超出范围：{index}（有效 1..{len(rows)}）"}, 400)
                    return
                row = rows[index - 1]
            else:
                row = random.choice(rows)
            try:
                reference = render_completion(row["completion"])
            except (KeyError, TypeError) as exc:
                self._send_json({"error": f"样本缺少 completion 字段：{exc}"}, 500)
                return
            self._send_json({
                "id": row.get("id"),
                "count": len(rows),
                "problem": row.get("problem") or {},
                "reference": reference,
            })

        # -------- 参数更新
        def _handle_config(self, body):
            new_stage = body.get("stage")
            if new_stage is not None and new_stage not in ("sft", "dpo"):
                self._send_json({"error": f"非法 stage：{new_stage!r}（应为 sft/dpo）"}, 400)
                return
            for key in _PARAM_KEYS:
                if key in body and body[key] is not None:
                    try:
                        value = int(body[key]) if key == "max_new_tokens" else float(body[key])
                    except (TypeError, ValueError):
                        self._send_json({"error": f"非法参数 {key}：{body[key]!r}"}, 400)
                        return
                    message = _validate_param(key, value)
                    if message:
                        self._send_json({"error": message}, 400)
                        return
                    setattr(state.args, key, value)
                    if state.session is not None:
                        setattr(state.session, key, value)

            if new_stage is not None and new_stage != state.args.stage:
                state.args.stage = new_stage
                state.args.adapter = body.get("adapter")  # 未显式指定则回到该阶段默认 adapter
                state.ready = False
                state.loading = True
                state.error = None
                print(f"[serve] 收到 stage 切换请求 -> {new_stage}，后台重载模型…", flush=True)
                threading.Thread(
                    target=_reload_worker,
                    args=(state, new_stage, state.args.adapter),
                    daemon=True,
                ).start()
            self._send_json(state.snapshot())

        # -------- SSE 生成
        def _handle_generate(self, body):
            problem = body.get("problem")
            if not isinstance(problem, dict) or not str(problem.get("question") or "").strip():
                self._send_json({"error": "「问题」为必填项，请先填写后再生成。"}, 400)
                return
            if not state.ready or state.session is None:
                self._send_json({"error": "模型尚未就绪，请稍候再试。"}, 400)
                return

            params = {}
            for key in _PARAM_KEYS:
                default = getattr(state.args, key)
                if key in body and body[key] is not None:
                    try:
                        default = int(body[key]) if key == "max_new_tokens" else float(body[key])
                    except (TypeError, ValueError):
                        self._send_json({"error": f"非法参数 {key}：{body[key]!r}"}, 400)
                        return
                params[key] = default

            # SSE 响应头：显式 no-cache；不发 Content-Encoding，即不做 gzip。
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True

            sse = _SSE(self.wfile)
            # 持锁串行化：显存有限，且模型非线程安全，并发请求排队而非并行。
            with state.gen_lock:
                if not state.ready or state.session is None:
                    sse.send("error", {"message": "模型尚未就绪，请稍候再试。"})
                    return
                session = state.session
                tokenizer = session.tokenizer
                try:
                    prompt = prompt_ids(tokenizer, problem)
                    if session.max_prompt_length and len(prompt) > session.max_prompt_length:
                        prompt = prompt[-session.max_prompt_length:]
                    null_prompt = prompt_ids(tokenizer, None)
                except Exception as exc:  # noqa: BLE001
                    sse.send("error", {"message": f"prompt 构造失败：{type(exc).__name__}: {exc}"})
                    return

                chunks = []

                def on_delta(delta):
                    chunks.append(delta)
                    sse.send("delta", {"text": "".join(chunks)})

                started = time.time()
                try:
                    generated = stream_generate(
                        session.model, tokenizer, prompt, null_prompt,
                        params["w"], params["max_new_tokens"],
                        params["temperature"], params["top_p"],
                        session.device, False, on_delta,
                    )
                except ClientGone:
                    print("[serve] 客户端已断开，生成中止。")
                    return
                except Exception as exc:  # noqa: BLE001
                    try:
                        sse.send("error", {"message": f"生成失败：{type(exc).__name__}: {exc}"})
                    except ClientGone:
                        pass
                    return

                elapsed = time.time() - started
                text = tokenizer.decode(generated, skip_special_tokens=True)
                summary = parse_labeled_output(text)
                try:
                    sse.send("done", {
                        "text": text,
                        "summary": summary,
                        "tokens": len(generated),
                        "seconds": round(elapsed, 3),
                    })
                except ClientGone:
                    pass

    return Handler


def _validate_param(key, value):
    """返回中文错误信息；合法返回 None。范围与前端滑块保持一致，防越界。"""
    if key == "w" and not 0.0 <= value <= 3.0:
        return f"w 需在 0.0~3.0 之间，收到 {value}"
    if key == "temperature" and not 0.0 < value <= 3.0:
        return f"temperature 需在 (0, 3.0] 之间，收到 {value}"
    if key == "top_p" and not 0.0 < value <= 1.0:
        return f"top_p 需在 (0, 1.0] 之间，收到 {value}"
    if key == "max_new_tokens" and value <= 0:
        return f"max_new_tokens 需为正整数，收到 {value}"
    return None


# ---------------------------------------------------------------- 入口
def _build_args(argv):
    cli = argparse.ArgumentParser(description="WIS 本地网页客户端后端（仅标准库）")
    cli.add_argument("--port", type=int, default=8765, help="监听端口，被占用则自动 +1")
    cli.add_argument("--host", default="127.0.0.1", help="监听地址")
    cli.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    cli.add_argument("--stage", choices=["sft", "dpo"], default="sft", help="使用哪一阶段的 adapter")
    cli.add_argument("--adapter", default=None, help="显式指定 adapter 目录")
    cli.add_argument("--config", default=None, help="配置文件路径，缺省按 stage 自动选择")
    opts = cli.parse_args(argv)
    # Session 期望的 args 属性集合，默认参数按实测调过（0.7/0.9 输出更通顺）。
    args = argparse.Namespace(
        stage=opts.stage, adapter=opts.adapter, config=opts.config,
        w=1.0, temperature=0.7, top_p=0.9, max_new_tokens=1600, max_prompt_length=512,
    )
    return args, opts


class _Server(ThreadingHTTPServer):
    # Windows 上 allow_reuse_address=1 会允许重复绑定已占用端口，使"自动 +1"失效；
    # 关掉它，端口被占时才会抛 OSError，从而走递增逻辑。
    allow_reuse_address = False
    daemon_threads = True


def _serve(host, port):
    """端口被占用则 +1 重试，返回可用端口。"""
    for offset in range(50):
        try:
            return _Server((host, port + offset), make_handler(_SERVE_STATE))
        except OSError:
            continue
    raise SystemExit(f"[serve] 无法绑定端口：{host}:{port}~{port + 49} 均被占用")


_SERVE_STATE = None


def main(argv=None):
    global _SERVE_STATE
    args, opts = _build_args(argv)
    state = AppState(args)
    _SERVE_STATE = state

    httpd = _serve(opts.host, opts.port)
    httpd.daemon_threads = True
    port = httpd.server_address[1]
    url = f"http://{opts.host}:{port}/"

    print("=" * 56, flush=True)
    print(f"[serve] 访问地址：{url}", flush=True)
    print(f"[serve] 阶段={args.stage}  配置={opts.config or '（按阶段自动选择）'}", flush=True)
    print("[serve] 正在后台加载模型（约 60 秒），加载完成后界面自动就绪…", flush=True)
    print("[serve] 按 Ctrl+C 退出。", flush=True)
    print("=" * 56, flush=True)

    threading.Thread(target=_load_worker, args=(state,), daemon=True).start()
    if opts.open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] 收到中断，正在退出…")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()