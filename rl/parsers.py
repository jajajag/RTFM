import re
from typing import Dict, List

def _split_list(s: str) -> List[str]:
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s+and\s+", ",", s, flags=re.I)
    return [x.strip() for x in s.split(",") if x.strip()]

def parse_instructions(text: str) -> Dict[str, List[str]]:
    """
    输入：RTFM 的说明文本（可能包含 facts + task）
    输出：{'tips': [句子...], 'goals': [句子...]}，并把并列项拆成多句
    """
    tips, goals = [], []
    sents = [s.strip() for s in re.split(r"[.\n]+", text) if s.strip()]

    for s in sents:
        s_norm = re.sub(r"\s*,\s*", ", ", s).strip()
        low = s_norm.lower()

        # 显式任务 / 动词命令 → goal
        if re.search(r"^\s*task\s*:?", low) or re.search(
            r"\b(slay|defeat|kill|reach|collect|bring|avoid|find|goal)\b", low
        ):
            goals.append(s_norm if s_norm.endswith(".") else s_norm + ".")
            continue

        # 阵营/成员/包含 → goal（并逐个实体拆开）
        m = re.match(r"^\s*([a-z\s,]+)\s+are\s+([a-z\s]+)\s*$", low)
        if m and re.search(r"\b(order|alliance|faction|enclave)\b", m.group(2)):
            members = _split_list(m.group(1))
            faction = m.group(2).strip()
            for mem in members:
                goals.append(f"{mem} are {faction}.")
            continue

        m = re.match(r"^\s*([a-z\s]+)\s+has\s+the\s+following\s+members\s*:\s*([a-z\s,]+)\s*$", low)
        if m:
            faction = m.group(1).strip()
            members = _split_list(m.group(2))
            for mem in members:
                goals.append(f"{mem} are {faction}.")
            continue

        m = re.match(r"^\s*([a-z\s]+)\s+contains\s+([a-z\s,]+)\s*$", low)
        if m:
            group = m.group(1).strip()
            members = _split_list(m.group(2))
            for mem in members:
                goals.append(f"{mem} are {group}.")
            continue

        # 规则类 → tip，并拆句
        m = re.match(r"^\s*([a-z\s]+)\s+monsters\s+are\s+defeated\s+by\s+([a-z\s,]+)\s+items?\s*$", low)
        if m:
            mon, items = m.group(1).strip(), _split_list(m.group(2))
            for it in items:
                tips.append(f"{mon} monsters are defeated by {it} items.")
            continue

        m = re.match(r"^\s*use\s+([a-z\s,]+)\s+items?\s+(?:to\s+beat|for)\s+([a-z\s]+)\s+monsters\s*$", low)
        if m:
            items, mon = _split_list(m.group(1)), m.group(2).strip()
            for it in items:
                tips.append(f"use {it} items to beat {mon} monsters.")
            continue

        m = re.match(r"^\s*([a-z\s,]+)\s+items?\s+are\s+good\s+against\s+([a-z\s]+)\s+monsters\s*$", low)
        if m:
            items, mon = _split_list(m.group(1)), m.group(2).strip()
            for it in items:
                tips.append(f"{it} items are good against {mon} monsters.")
            continue

        # 默认当 tip
        tips.append(s_norm if s_norm.endswith(".") else s_norm + ".")

    return {"tips": tips, "goals": goals}

