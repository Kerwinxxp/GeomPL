"""Eq.8 状态 St = {Ht, Vt, Mt, Ct}，全 JSON 可序列化。  [paper]

Ht: {location: posterior}；Vt: 任务列表（Eq.9 四字段）；
Mt: key evidence 列表；Ct: 推理上下文（层级、历史、事件）。
"""
from dataclasses import dataclass, field


@dataclass
class State:
    hypotheses: dict = field(default_factory=dict)          # Ht
    plan: list = field(default_factory=list)                # Vt: [{desc,reason,bbox,status}]
    memory: list = field(default_factory=list)              # Mt: key evidence
    context: dict = field(default_factory=dict)             # Ct: {level, history, ...}

    def to_dict(self) -> dict:
        return {
            "hypotheses": dict(self.hypotheses),
            "plan": [dict(t) for t in self.plan],
            "memory": list(self.memory),
            "context": dict(self.context),
        }
