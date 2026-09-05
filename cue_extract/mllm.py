"""GPT-4o 客户端(OpenAI 兼容端点),供线索提取使用。

- transport 可注入（Callable[[messages], str]），便于离线测试；默认走 openai SDK。
- 磁盘缓存：key = sha256(model + temperature + messages摘要)，图像以其 data-url 的
  sha256 入 key；缓存格式与 key 计算与旧 geobayes.mllm.client 完全一致,
  因此 .mllm_cache/ 中已付费的响应继续命中。
- JSON 解析失败重试（附错误提示）。
- 图像本地 smart_resize 后上传：模型坐标空间 == 我方像素空间。
- API key **只**从环境变量 OPENAI_API_KEY 读取,绝不写进代码或配置。
"""
import base64
import hashlib
import io
import json
import os
import re

from .imaging import smart_resize_dims

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict:
    """从模型回复中提取 JSON **对象**：裸 JSON / markdown 围栏 / 前后缀噪声 / 数组包裹。

    只接受 dict——数组或标量顶层视为无效（防止污染缓存后在下游炸掉）。
    """
    for candidate in (text, *_FENCE_RE.findall(text)):
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no valid JSON object in model reply: {text[:200]!r}")


class MLLMClient:
    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str | None = None, cache_dir: str | None = None,
                 transport=None, temperature: float = 0.0,
                 json_retries: int = 2, max_pixels: int = 1280 * 28 * 28):
        self.model = model
        self.temperature = temperature
        self.json_retries = json_retries
        self.max_pixels = max_pixels
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._transport = transport or self._default_transport(api_key, base_url)

    def prepare(self, image):
        """smart_resize 到模型坐标空间。公开：流水线入口调用一次，使全流程
        （掩码 / bbox）与模型看到的像素空间一致；本方法幂等。"""
        w, h = smart_resize_dims(image.width, image.height,
                                 max_pixels=self.max_pixels)
        if (w, h) != image.size:
            image = image.resize((w, h))
        return image

    def vision_json(self, prompt: str, image) -> dict:
        """通用图像→JSON 原语:预处理图像,用 {width}/{height} 格式化 prompt,返回解析后的 JSON。"""
        image = self.prepare(image)
        try:
            prompt = prompt.format(width=image.width, height=image.height)
        except (KeyError, IndexError):
            pass
        return self._chat_json([self._vision_msg(prompt, image)])

    # ---------- 内部 ----------

    def _chat_json(self, messages: list) -> dict:
        cached = self._cache_get(messages)
        if cached is not None:
            return cached
        attempt_msgs = list(messages)
        last_err = None
        for _ in range(1 + self.json_retries):
            reply = self._transport(attempt_msgs)
            try:
                result = extract_json(reply)
                self._cache_put(messages, result)
                return result
            except ValueError as err:
                last_err = err
                attempt_msgs = attempt_msgs + [
                    {"role": "assistant", "content": str(reply)[:2000]},
                    {"role": "user",
                     "content": "Your previous reply was not valid JSON. "
                                "Reply again with ONLY the JSON object."},
                ]
        raise ValueError(f"model did not return valid JSON after retries: {last_err}")

    def _vision_msg(self, text: str, image) -> dict:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": text},
        ]}

    def _cache_key(self, messages: list) -> str:
        def strip_images(obj):
            if isinstance(obj, dict):
                return {k: strip_images(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [strip_images(v) for v in obj]
            if isinstance(obj, str) and obj.startswith("data:image"):
                return hashlib.sha256(obj.encode()).hexdigest()
            return obj
        payload = json.dumps({"model": self.model, "temperature": self.temperature,
                              "messages": strip_images(messages)},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_get(self, messages):
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, self._cache_key(messages) + ".json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _cache_put(self, messages, result):
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, self._cache_key(messages) + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

    def _default_transport(self, api_key, base_url):
        def transport(messages):
            try:
                from openai import OpenAI
            except ImportError as err:
                raise RuntimeError(
                    "pip install openai, or inject a custom transport"
                ) from err
            client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url or "https://api.openai.com/v1",
            )
            resp = client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature
            )
            return resp.choices[0].message.content
        return transport
