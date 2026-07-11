"""真实 MLLM 客户端（OpenAI 兼容端点）。

- transport 可注入（Callable[[messages], str]），便于离线测试；默认走 openai SDK。
- 磁盘缓存：key = sha256(model + messages摘要)，图像以其字节哈希入 key（map §4）。
- JSON 解析失败重试（附错误提示），judge 缺失标签兜底 (c=3, alpha=0) → W=1。
- 图像本地 smart_resize 后上传：模型坐标空间 == 我方像素空间。
"""
import base64
import hashlib
import io
import json
import os
import re

from ..eval.geocode import canonicalize_country
from . import prompts
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

    # ---------- 四个调用 ----------

    def hypothesize(self, image, k: int = 5, max_tasks: int = 6,
                    granularity: str = "country") -> dict:
        image = self.prepare(image)
        spec = prompts.GRANULARITY.get(granularity, prompts.GRANULARITY["country"])
        prompt = prompts.HYPOTHESIZE.format(
            k=k, max_tasks=max_tasks, width=image.width, height=image.height,
            level=granularity, unit=spec["unit"],
            unit_plural=spec["unit_plural"], example=spec["example"])
        result = self._chat_json([self._vision_msg(prompt, image)])
        result["level"] = granularity   # 以配置的粒度为准，不信任模型自报
        # 仅国家层做国家名规范化；城市/地点候选保持原样
        result["candidates"] = self._clean_candidates(result.get("candidates"), granularity)
        if not result["candidates"]:
            raise ValueError("hypothesize returned no candidates")
        return result

    def score_candidates(self, image, candidate_labels, smoothing: float = 0.02) -> dict:
        """闭集打分 → 先验分布。对每个给定候选独立打分 [0,1]，归一化 + 轻度平滑
        （保证闭集内无硬零，贝叶斯更新可恢复任一候选）。返回 {prior, raw_scores}。"""
        labels = list(candidate_labels)
        image = self.prepare(image)
        prompt = prompts.SCORE_CANDIDATES.format(
            candidates=json.dumps(labels, ensure_ascii=False))
        result = self._chat_json([self._vision_msg(prompt, image)])
        scores = result.get("scores", {})
        raw = {l: min(1.0, max(0.0, float(scores.get(l, 0.0) or 0.0))) for l in labels}
        total = sum(raw.values())
        n = len(labels)
        base = {l: (raw[l] / total if total > 0 else 1.0 / n) for l in labels}
        prior = {l: (1 - smoothing) * base[l] + smoothing / n for l in labels}
        z = sum(prior.values())
        prior = {l: prior[l] / z for l in labels}
        return {"prior": prior, "raw_scores": raw}

    def plan_verification(self, image, candidate_labels, max_tasks: int = 6) -> dict:
        """规划用于区分给定候选的验证任务（含 bbox）。"""
        image = self.prepare(image)
        prompt = prompts.PLAN_VERIFICATION.format(
            candidates=json.dumps(list(candidate_labels), ensure_ascii=False),
            max_tasks=max_tasks, width=image.width, height=image.height)
        result = self._chat_json([self._vision_msg(prompt, image)])
        result.setdefault("verification_tasks", [])
        return result

    def vision_json(self, prompt: str, image) -> dict:
        """通用图像→JSON 原语:预处理图像,用 {width}/{height} 格式化 prompt,返回解析后的 JSON。

        供 clue_leak 等外部研究模块复用(检测线索框等),不绑定具体任务。
        """
        image = self.prepare(image)
        try:
            prompt = prompt.format(width=image.width, height=image.height)
        except (KeyError, IndexError):
            pass
        return self._chat_json([self._vision_msg(prompt, image)])

    def zero_shot(self, image) -> dict:
        """论文 baseline：单次调用直接给层级地名（无贝叶斯框架、无搜索、无验证循环）。"""
        from ..eval.geocode import assemble_name
        image = self.prepare(image)
        result = self._chat_json([self._vision_msg(prompts.ZERO_SHOT, image)])
        parts = [result.get("street"), result.get("city"), result.get("country")]
        result["name"] = assemble_name(parts)
        return result

    def verify(self, crop_image, task: dict) -> dict:
        crop_image = self.prepare(crop_image)
        prompt = prompts.VERIFY.format(desc=task.get("desc", ""),
                                       reason=task.get("reason", ""))
        result = self._chat_json([self._vision_msg(prompt, crop_image)])
        result.setdefault("observation", "")
        result.setdefault("geo_clues", [])
        return result

    def judge(self, evidence_text: str, labels: list) -> dict:
        prompt = prompts.JUDGE.format(evidence=evidence_text,
                                      labels=json.dumps(list(labels)))
        messages = [{"role": "user", "content": prompt}]
        result = self._chat_json(messages)
        ratings_raw = dict(result.get("ratings", {}))
        missing = [l for l in labels if l not in ratings_raw]
        extra = [k for k in ratings_raw if k not in labels]
        if missing or extra:  # 缺失/多余均触发一次针对性重试（map §3.3）
            hint = (f"Your ratings object must contain exactly these hypotheses, "
                    f"each exactly once: {list(labels)}. "
                    f"Missing: {missing}. Unexpected: {extra}. "
                    f"Return the corrected full JSON.")
            try:
                retry = self._chat_json(messages + [
                    {"role": "assistant", "content": json.dumps(result)},
                    {"role": "user", "content": hint},
                ])
                # 合并而非替换：重试只补缺口时，首轮有效打分必须保留
                ratings_raw = {**ratings_raw, **retry.get("ratings", {})}
            except ValueError:
                pass  # 重试对话失败 → 降级走兜底，不炸掉整个 run
            missing = [l for l in labels if l not in ratings_raw]
            extra = [k for k in ratings_raw if k not in labels]
        ratings = {l: ratings_raw[l] for l in labels if l in ratings_raw}
        for l in missing:  # 兜底：中性 → W=1（map §3.3）
            ratings[l] = {"c": 3, "alpha": 0.0}
        out = {"ratings": ratings}
        if missing:
            out["fallback_labels"] = sorted(missing)
        if extra:
            out["extra_labels"] = sorted(extra)
        return out

    def subhypotheses(self, parent, level, objects, snippets) -> dict:
        snippet_txt = "\n".join(
            f"- {s.get('title','')}: {s.get('content','')}" for s in (snippets or [])
        ) or "(no results)"
        prompt = prompts.SUBHYPOTHESES.format(
            parent=parent, level=level,
            objects=json.dumps(list(objects), ensure_ascii=False),
            snippets=snippet_txt[:2500])
        result = self._chat_json([{"role": "user", "content": prompt}])
        # 下层（city/street）不做国家规范化，仅去重清洗
        result["candidates"] = self._clean_candidates(result.get("candidates"), level)
        if not result["candidates"]:
            raise ValueError("subhypotheses returned no candidates")
        return result

    def replace(self, context: dict) -> dict:
        prompt = prompts.REPLACE.format(
            scene_summary=context.get("scene_summary", ""),
            failed=json.dumps(context.get("failed_hypotheses", [])),
            memory=json.dumps(context.get("memory", []))[:2000],
            level=context.get("level", "country"),
        )
        result = self._chat_json([{"role": "user", "content": prompt}])
        result["candidates"] = self._clean_candidates(
            result.get("candidates"), result.get("level", context.get("level", "country"))
        )
        if not result["candidates"]:
            raise ValueError("replace returned no candidates")
        return result

    # ---------- 内部 ----------

    def _clean_candidates(self, candidates, level: str) -> list:
        """map §3.1 校验：country 层 canonicalize；去重取最大 si；si 钳制 [0,1]。"""
        if not candidates:
            return []
        cleaned, order = {}, []
        for c in candidates:
            loc = str(c.get("location", "")).strip()
            if not loc:
                continue
            if level == "country":
                loc = canonicalize_country(loc)
            try:
                conf = min(1.0, max(0.0, float(c.get("confidence", 0.0))))
            except (TypeError, ValueError):
                conf = 0.0
            if loc in cleaned:
                cleaned[loc]["confidence"] = max(cleaned[loc]["confidence"], conf)
            else:
                entry = {"location": loc, "confidence": conf}
                if c.get("reason"):
                    entry["reason"] = c["reason"]
                cleaned[loc] = entry
                order.append(loc)
        return [cleaned[l] for l in order]

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

    def prepare(self, image):
        """smart_resize 到模型坐标空间。公开：controller 在流水线入口调用一次，
        使全流程（bbox 裁剪）与模型看到的像素空间一致；本方法幂等。"""
        w, h = smart_resize_dims(image.width, image.height,
                                 max_pixels=self.max_pixels)
        if (w, h) != image.size:
            image = image.resize((w, h))
        return image

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
                api_key=api_key or os.environ.get("DASHSCOPE_API_KEY")
                or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url
                or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            resp = client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature
            )
            return resp.choices[0].message.content
        return transport
