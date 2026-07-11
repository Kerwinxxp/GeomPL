"""合并/质检逻辑(PLAN §2②③⑤):IoU 去重、OCR 并入、多实例归组、退化框标记。"""


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def group_instances(cue: dict, detections: list) -> dict:
    """一条 proposal 线索 + grounding 检出的全部框 → 带实例列表的线索。

    多实例保留全部(将来遮蔽消融必须 union 遮,PLAN §2② 多实例规则)。
    """
    return {**cue, "instances": [
        {"bbox": [float(v) for v in d["bbox"]], "score": float(d.get("score", 0.0)),
         "source": "grounding"} for d in detections]}


def merge_ocr_into_cues(cues: list, ocr_results: list, iou_thr: float = 0.5) -> list:
    """OCR 文字框并入线索列表:与已有实例 IoU>阈值 → 把文本挂到该实例;
    否则新建一条 text/signage 线索(source=ocr)。"""
    out = [dict(c, instances=[dict(i) for i in c.get("instances", [])]) for c in cues]
    for o in ocr_results:
        hit = None
        for c in out:
            for inst in c["instances"]:
                if iou(inst["bbox"], o["bbox"]) > iou_thr:
                    hit = inst
                    break
            if hit:
                break
        if hit is not None:
            hit["text"] = o["text"]
            hit["ocr_conf"] = float(o.get("conf", 0.0))
        else:
            out.append({"cue": f"text: {o['text'][:40]}", "category": "text/signage",
                        "grounding_phrase": "text", "is_text": True, "source": "ocr",
                        "text": o["text"],
                        "instances": [{"bbox": [float(v) for v in o["bbox"]],
                                       "score": float(o.get("conf", 0.0)), "source": "ocr"}]})
    return out


def prune_uncorroborated_text_boxes(cues: list) -> list:
    """问题①修复:文字线索(is_text)的 grounding 框必须有 OCR 佐证(实例带 text 或 source=ocr),
    否则视为幻觉框剔除——grounding 对 'website URL'/'hotel sign' 这类抽象短语常乱框。
    非文字线索不动。"""
    out = []
    for c in cues:
        if c.get("is_text") and c.get("source") != "ocr":
            kept = [i for i in c.get("instances", [])
                    if i.get("text") or i.get("source") == "ocr"]
            out.append(dict(c, instances=kept))
        else:
            out.append(c)
    return out


def assign_maskable(cues: list) -> list:
    """问题③修复:maskable 由证据决定,不由 LLM 主观判——有任一"非退化的实际框"即可遮
    (治 NYC 天际线明明有好 mask 却被判 global)。verifier 的语义意见留存到 maskable_llm。"""
    out = []
    for c in cues:
        has_box = any(not i.get("degenerate") for i in c.get("instances", []))
        d = dict(c)
        if "maskable" in d:
            d["maskable_llm"] = d["maskable"]
        d["maskable"] = bool(has_box)
        out.append(d)
    return out


def flag_degenerate(cues: list, image_size, max_ratio: float = 0.4) -> list:
    """实例级退化标记:bbox 面积 > max_ratio×全图 的实例标 degenerate(不做 mask/遮蔽);
    线索级 degenerate = 全部实例都退化(好实例不被坏实例连坐)。(PLAN §2⑤ 硬性职责 1)"""
    w, h = image_size
    total = float(w * h)
    out = []
    for c in cues:
        insts = [dict(i, degenerate=(i["bbox"][2] - i["bbox"][0]) *
                      (i["bbox"][3] - i["bbox"][1]) / total > max_ratio)
                 for i in c.get("instances", [])]
        out.append(dict(c, instances=insts,
                        degenerate=bool(insts) and all(i["degenerate"] for i in insts)))
    return out
