"""线索质检:退化框标记 + 证据驱动的可遮判定(route-B + SAM3 管线)。"""

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
