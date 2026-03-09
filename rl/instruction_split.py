from typing import List


def split_instructions(paragraph: str) -> List[str]:
    paragraph = (paragraph or "").strip()
    if not paragraph:
        return ["(no instruction)"]

    sents = [s.strip() for s in paragraph.replace("\n", " ").split(".") if s.strip()]
    if not sents:
        return [paragraph]

    ret: List[str] = []
    for sent in sents:
        words = sent.split()
        if not words:
            continue
        first_half, second_half = "", ""
        half, middle = [], set()
        i = 0
        while i < len(words):
            if words[i] == ",":
                if not first_half and len(half) > 1:
                    first_half = " ".join(half[:-1])
                if i > 0:
                    middle.add(words[i - 1])
                if i + 1 < len(words):
                    middle.add(words[i + 1])
                half = []
            half.append(words[i])
            i += 1
        second_half = " ".join(half[2:]) if len(half) > 2 else " ".join(half)
        for word in middle:
            tip = f"{first_half} {word} {second_half}".strip()
            if tip:
                ret.append(tip)

    return ret or sents


def split_with_parser(paragraph: str, *_, **__) -> List[str]:
    return split_instructions(paragraph)


def split_with_lm(paragraph: str, *_, **__) -> List[str]:
    paragraph = (paragraph or "").strip()
    sents = [s.strip() for s in paragraph.replace("\n", " ").split(".") if s.strip()]
    return sents or ["(no instruction)"]
