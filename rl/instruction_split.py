# rl/instruction_split.py
from typing import List

def split_instructions(paragraph: str) -> List[str]:
    paragraph = (paragraph or "").strip()
    if not paragraph:
        return ["(no instruction)"]

    sents = paragraph.split('.')

    ret = []

    for sent in sents:
        words = sent.split()
        if not words:
            continue
        first_half, second_half = '', ''
        half, middle = [], set()
        i = 0
        while i < len(words):
            if words[i] == ',':
                if not first_half:
                    first_half = ' '.join(half[:-1])
                middle.add(words[i - 1])
                middle.add(words[i + 1])
                half = []
            half.append(words[i])
            i += 1
        second_half = ' '.join(half[2:])
        for word in middle:
            ret.append(f'{first_half} {word} {second_half}'.strip())
    return ret

# Backward-compatible names
def split_with_parser(paragraph: str, *_, **__) -> List[str]:
    return split_instructions(paragraph)

def split_with_lm(paragraph: str, *_, **__) -> List[str]:
    sents = paragraph.split('.')
    ret = [sent.strip() for sent in sents if sent]
    return ret

if __name__ == '__main__':
    data = 'rebel enclave contains ghost , jaguar , wolf . cold monsters are defeated by grandmasters , soldiers items . you should use mysterious , shimmering items to beat poison monsters . goblin , imp , shaman are order of the forest . blessed , gleaming items are good against fire monsters . star alliance has the following members : bat , panther , zombie . use arcane , fanatical items for lightning monsters .'
    print(split_with_parser(data))
    print(split_with_lm(data))
