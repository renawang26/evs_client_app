"""
word_aligner - Streamlit custom component for EN↔ZH word alignment.

Usage:
    from word_aligner import word_aligner

    pairs = word_aligner(
        en_words=["The", "gender", "inequality"],
        zh_words=["性别", "不平等"],
    )
    if pairs:
        st.json(pairs)
"""

import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

_component_func = components.declare_component(
    "word_aligner",
    path=_COMPONENT_DIR,
)


def word_aligner(
    en_words: list[str],
    zh_words: list[str],
    default_pairs: list[dict] | None = None,
    height: int = 0,
    key: str | None = None,
) -> list[dict] | None:
    """Render an interactive EN↔ZH word alignment widget.

    Parameters
    ----------
    en_words : list[str]
        English tokens to display in the top row.
    zh_words : list[str]
        Chinese tokens to display in the bottom row.
    default_pairs : list[dict], optional
        Pre-existing pairs to restore, each dict must have
        ``en``, ``zh``, ``en_idx``, ``zh_idx`` keys.
    height : int, optional
        Fixed iframe height in px.  0 = auto-size (default).
    key : str, optional
        Streamlit widget key for state management.

    Returns
    -------
    list[dict] | None
        A list of pair dicts when the user clicks "Save Pairs",
        or ``None`` if nothing has been saved yet.
        Each dict: ``{"en": str, "zh": str, "en_idx": int, "zh_idx": int}``
    """
    result = _component_func(
        en_words=en_words,
        zh_words=zh_words,
        default_pairs=default_pairs or [],
        height=height,
        key=key,
        default=None,
    )
    return result
