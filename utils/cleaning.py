import re

def clean_text(text: str) -> str:
    # remove citations like [31], [31, 2, 8]
    text = re.sub(r"\[\d+(,\s*\d+)*\]", "", text)

    # remove double spaces
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()