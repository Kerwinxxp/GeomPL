"""【实验性 · 可整体删除】GeoRanker 作为信念计:对固定坐标 gallery 输出连续 softmax 概率。

模型 = Qwen2-VL-7B + value_head(线性,取末 token 隐状态 → 标量 reward,越高=越近)
     + 官方 LoRA(belief_elicit/georanker_ckpt/,来源 commit 见 SOURCE_COMMIT.txt)。

不 import 官方 utils/geo_ranker.py(其顶层 import flash_attn/deepspeed,Windows 装不上);
RewardModel 逻辑按官方文件逐行复刻(~40 行),attn 用 sdpa 替代 flash_attention_2。

prompt 变体(Okazaki 体检对比后定标准;负例若用,须从【原图】算一次后对该图所有
遮蔽条件复用,保证 original/masked 的 prompt 恒定可比):
  A  gps      : query图 + "How far is this place from latitude: X, longitude: Y?"
  B  gps+text : 同上,但坐标后接候选文字("Kyoto, Japan")
  C  B + negatives : 再附 "Negative examples: lat,lon,text; ..."(训练格式,减参考图)

运行环境:belief_elicit/.venv_gr(torch 2.11+cu128 / transformers 4.5x / peft)。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import base64
import io

import torch
import torch.nn as nn

CKPT = os.path.join(os.path.dirname(__file__), "georanker_ckpt")
BASE = "Qwen/Qwen2-VL-7B-Instruct"
_MODEL, _PROC = None, None


# ---------- RewardModel(复刻官方 utils/geo_ranker.py 的 _get_reward_model) ----------

def _reward_model_cls():
    from transformers import Qwen2VLForConditionalGeneration

    class RewardModel(Qwen2VLForConditionalGeneration):
        def __init__(self, config):
            super().__init__(config)
            self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

        def forward(self, input_ids, attention_mask=None, pixel_values=None,
                    return_output=False, **kwargs):
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                      pixel_values=pixel_values, output_hidden_states=True,
                                      **kwargs)
            last_hidden = outputs.hidden_states[-1]
            values = self.value_head(last_hidden).squeeze(-1)
            return values[:, -1]                      # 末 token(左 padding 下对齐)

    return RewardModel


def get_model(quant="nf4"):
    """quant='nf4'(默认,~4.5GB,17GB 卡不换页)| 'bf16'(15GB+,会溢出到系统内存,极慢)。"""
    global _MODEL, _PROC
    if _MODEL is None:
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig
        cls = _reward_model_cls()
        kw = dict(attn_implementation="sdpa", device_map="cuda")
        if quant == "nf4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["value_head", "lm_head"])   # 打分头保持全精度
        else:
            kw["dtype"] = torch.bfloat16
        m = cls.from_pretrained(BASE, **kw)
        m = PeftModel.from_pretrained(m, CKPT, is_trainable=False)   # 显式加载,绕开 auto_mapping
        m.eval()
        # 验证 LoRA 的 value_head(modules_to_save)真的替换了随机初始化头
        vh = m.base_model.model.value_head
        assert hasattr(vh, "modules_to_save"), "value_head 未被 PEFT 包裹 —— adapter 未生效!"
        _PROC = AutoProcessor.from_pretrained(BASE, use_fast=True)
        _PROC.tokenizer.padding_side = "left"         # reward 取末位,须左 padding
        _MODEL = m
    return _MODEL, _PROC


# ---------- prompt 构造 ----------

def _img_str(pil_image):
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_message(img_str, gps, text=None, negatives=None):
    q = f"How far is this place from latitude: {gps[0]}, longitude: {gps[1]}"
    q += f", {text}?" if text else "?"
    content = [{"type": "image", "image": img_str}, {"type": "text", "text": q}]
    if negatives:
        content.append({"type": "text", "text": "Negative examples: " + "; ".join(negatives)})
    return [{"role": "user", "content": content}]


def format_negatives(neg_entries):
    """neg_entries: [(lat, lon, text), ...] → 训练格式字符串列表。"""
    return [f"latitude: {la}, longitude: {lo}, {tx}" for la, lo, tx in neg_entries]


# ---------- 打分 ----------

@torch.no_grad()
def score_rewards(pil_image, cand_gps, cand_texts=None, negatives=None, batch_size=8):
    """对一张图 × N 候选打 reward。cand_texts=None → 变体A;有 texts → B;再有 negatives → C。
    返回 list[float](与 cand_gps 同序)。"""
    from qwen_vl_utils import process_vision_info
    model, proc = get_model()
    img_str = _img_str(pil_image)
    msgs = [build_message(img_str, g,
                          text=(cand_texts[i] if cand_texts else None),
                          negatives=negatives)
            for i, g in enumerate(cand_gps)]
    rewards = []
    for i in range(0, len(msgs), batch_size):
        chunk = msgs[i:i + batch_size]
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in chunk]
        image_inputs, video_inputs = process_vision_info(chunk)
        inputs = proc(text=texts, images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors="pt").to("cuda")
        r = model(**inputs)
        rewards.extend(r.float().cpu().tolist())
    return rewards


def softmax_probs(rewards, tau=1.0):
    t = torch.tensor(rewards, dtype=torch.float64) / tau
    p = torch.softmax(t, dim=0).tolist()
    return p


def score_labels(pil_image, gallery, variant="B", negatives=None, tau=1.0, batch_size=8):
    """gallery: gallery_v2 记录列表 [{label, gps, ...}](须有 gps)→ {label: prob}。
    variant: 'A' 仅gps | 'B' gps+label文字 | 'C' B+negatives(调用方备好并跨条件复用)。"""
    gv = [g for g in gallery if g.get("gps")]
    gps = [g["gps"] for g in gv]
    texts = [g["label"] for g in gv] if variant in ("B", "C") else None
    negs = negatives if variant == "C" else None
    rewards = score_rewards(pil_image, gps, cand_texts=texts, negatives=negs,
                            batch_size=batch_size)
    probs = softmax_probs(rewards, tau=tau)
    return {gv[i]["label"]: probs[i] for i in range(len(gv))}, rewards
