"""Coordinate parsing and character-span to generated-token importance mapping."""

import re

PAIR = re.compile(r"\(\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)")


def parse_point(text):
    match = PAIR.fullmatch(text.strip())
    if match is None:
        return None
    x, y = map(int, match.groups())
    return (x / 1000, y / 1000) if 0 <= x <= 1000 and 0 <= y <= 1000 else None


def token_importance(tokenizer, token_ids, base_weight=1.0, digit_scale=1.0):
    """Use actual generated IDs, never re-tokenize and assume identical boundaries.

    A digit has weight base + scale * place_value/1000. A merged token takes
    the maximum weight of its characters. Invalid answers keep base weights.
    """
    if base_weight <= 0 or digit_scale < 0:
        raise ValueError("base_weight must be positive; digit_scale nonnegative")
    ids = list(token_ids)

    def decode(xs):
        return tokenizer.decode(xs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    text = decode(ids)
    spans, previous = [], ""
    for i in range(len(ids)):
        current = decode(ids[: i + 1])
        if not current.startswith(previous):
            # Defensive fallback for non-prefix-stable Unicode tokenizers.
            return [base_weight] * len(ids)
        spans.append((len(previous), len(current)))
        previous = current
    result = [base_weight] * len(ids)
    stripped = text.strip()
    match = PAIR.fullmatch(stripped)
    if match is None or parse_point(stripped) is None:
        return result
    offset = len(text) - len(text.lstrip())
    char_weights = {}
    for group in (1, 2):
        start, end = match.span(group)
        for pos in range(start, end):
            char_weights[offset + pos] = base_weight + digit_scale * 10 ** (end - pos - 1) / 1000
    for i, (start, end) in enumerate(spans):
        result[i] = max([base_weight] + [char_weights.get(j, base_weight) for j in range(start, end)])
    return result
