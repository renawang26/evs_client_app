"""
evs_annotator - Streamlit custom component for EVS word annotation.

Replaces the Compact (Buttons) mode in Annotate EVS tab with a
client-side HTML/JS component that handles selection without
triggering Streamlit reruns.
"""

import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

_component_func = components.declare_component(
    "evs_annotator",
    path=_COMPONENT_DIR,
)


def evs_annotator(
    groups: list[dict],
    seconds_per_row: int = 5,
    font_size: int = 14,
    en_selections: dict | None = None,
    zh_selections: dict | None = None,
    height: int = 0,
    key: str | None = None,
) -> dict | None:
    """Render an interactive EVS word annotation widget.

    All word selection happens client-side in JS — no Streamlit rerun
    until the component sends data back.

    Parameters
    ----------
    groups : list[dict]
        Time groups, each with keys:
        - ``time_group_key`` (str): e.g. "00:30"
        - ``words`` (list[dict]): each word dict has keys
          ``ts``, ``ts_float``, ``en`` (dict|None), ``zh`` (dict|None).
          The en/zh dicts have ``word``, ``segment_id``, ``word_seq_no``,
          ``start_time``, ``pair_seq``, ``pair_type``.
    seconds_per_row : int
        Seconds of audio per visual row.
    font_size : int
        Font size in px for word buttons.
    en_selections : dict, optional
        Current English selections to restore (key -> info dict).
    zh_selections : dict, optional
        Current Chinese selections to restore.
    height : int, optional
        Fixed iframe height in px. 0 = auto-size (default).
    key : str, optional
        Streamlit widget key for state management.

    Returns
    -------
    dict | None
        When selections change the component returns::

            {
                "action": "selections_changed",
                "en_selections": { key: {time, word, segment_id, word_seq_no}, ... },
                "zh_selections": { ... }
            }

        Returns ``None`` on first render before any interaction.
    """
    result = _component_func(
        groups=groups,
        seconds_per_row=seconds_per_row,
        font_size=font_size,
        en_selections=en_selections or {},
        zh_selections=zh_selections or {},
        height=height,
        key=key,
        default=None,
    )
    return result
