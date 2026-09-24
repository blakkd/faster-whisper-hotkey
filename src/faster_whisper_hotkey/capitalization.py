"""Capitalization post-processing for models that emit no capitalization.

granite-speech-4.1-2b-nar produces punctuation but all-lowercase text.
This module restores basic sentence case without touching the model's
wording. granite-turboctc is intentionally not post-processed: it emits
no punctuation, so sentence boundaries cannot be detected reliably (use
LLM correction instead).
"""

import re

# Standalone "i" is the English first-person pronoun. Also covers
# contractions like "i'm" and "i'll" (the apostrophe is a word boundary).
_STANDALONE_I_RE = re.compile(r"\bi\b")

# Closing quotes/brackets that may follow a sentence-ending mark.
_CLOSERS = "\"'”’)]»"


def add_capitalization(text: str, language: str = "en") -> str:
    """Capitalize sentence starts, and standalone "i" when language is English.

    A sentence start is the beginning of the text, or the first letter
    after ``.``/``!``/``?`` (optionally followed by closing quotes) and
    whitespace. Caseless scripts (CJK) and already-capitalized text are
    left untouched.
    """
    if not text:
        return text

    chars = list(text)

    def capitalize_first_alpha(start: int) -> None:
        i = start
        while i < len(chars) and not chars[i].isalpha():
            i += 1
        if i < len(chars):
            # "ß".upper() would yield "SS"; a German word starting with ß
            # (only the straße family) capitalizes to "Straße".
            chars[i] = "Straße" if chars[i] == "ß" else chars[i].upper()

    capitalize_first_alpha(0)

    for i, ch in enumerate(chars):
        if ch in ".!?":
            j = i + 1
            while j < len(chars) and chars[j] in _CLOSERS:
                j += 1
            if j < len(chars) and chars[j].isspace():
                capitalize_first_alpha(j)

    text = "".join(chars)
    if language == "en":
        text = _STANDALONE_I_RE.sub("I", text)
    return text
