"""
Download Audio tab - Download audio from YouTube using yt-dlp.

Supports multi-track audio (e.g., UN conference videos with floor + interpretation tracks)
and single-stream videos with stereo L/R channel separation.
"""

import streamlit as st
import os
import re
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

EVS_RESOURCES_PATH = "./evs_resources"
DOWNLOADS_DIR = os.path.join(EVS_RESOURCES_PATH, "downloads")


def _ensure_download_dir():
    """Ensure the downloads directory exists."""
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)


def _sanitize_filename(title: str) -> str:
    """Create a safe filename from video title + timestamp."""
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '', title)
    safe = re.sub(r'\s+', '_', safe.strip())
    safe = safe[:80]  # Limit length
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe}_{timestamp}"


def _fetch_video_info(url: str) -> dict:
    """Fetch video metadata without downloading."""
    import yt_dlp

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info


def _classify_audio_formats(formats: list) -> dict:
    """
    Classify audio formats, detecting multi-track (language-tagged) streams.

    Returns:
        {
            'is_multi_track': bool,
            'tracks': {language_tag: [format_dicts]},
            'best_audio': format_dict or None
        }
    """
    audio_formats = [f for f in formats if f.get('acodec', 'none') != 'none' and f.get('vcodec', 'none') == 'none']

    tracks_by_lang = {}
    for fmt in audio_formats:
        lang = fmt.get('language') or 'und'
        if lang not in tracks_by_lang:
            tracks_by_lang[lang] = []
        tracks_by_lang[lang].append(fmt)

    # Sort each language's formats by quality (abr or filesize)
    for lang in tracks_by_lang:
        tracks_by_lang[lang].sort(
            key=lambda f: f.get('abr') or f.get('filesize') or 0,
            reverse=True
        )

    # Multi-track if more than one distinct language tag (excluding 'und')
    real_langs = [l for l in tracks_by_lang if l != 'und']
    is_multi_track = len(real_langs) >= 2

    # Find best overall audio format
    best_audio = None
    if audio_formats:
        best_audio = max(audio_formats, key=lambda f: f.get('abr') or 0)

    return {
        'is_multi_track': is_multi_track,
        'tracks': tracks_by_lang,
        'best_audio': best_audio,
    }


def _get_language_label(lang_code: str) -> str:
    """Convert language code to human-readable label."""
    labels = {
        'en': 'English',
        'zh': 'Chinese (中文)',
        'zh-hans': 'Chinese Simplified (简体中文)',
        'zh-hant': 'Chinese Traditional (繁体中文)',
        'fr': 'French (Français)',
        'es': 'Spanish (Español)',
        'ar': 'Arabic (العربية)',
        'ru': 'Russian (Русский)',
        'und': 'Unknown Language',
    }
    return labels.get(lang_code, lang_code)


def _format_duration(seconds) -> str:
    """Format seconds into HH:MM:SS."""
    if not seconds:
        return "Unknown"
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _download_audio(url: str, format_id: str, output_path: str, progress_placeholder) -> str:
    """Download a specific audio format and convert to WAV."""
    import yt_dlp

    final_path = output_path
    if not final_path.endswith('.wav'):
        final_path = os.path.splitext(output_path)[0] + '.wav'

    def progress_hook(d):
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
            downloaded = d.get('downloaded_bytes', 0)
            if total > 0:
                pct = downloaded / total
                progress_placeholder.progress(pct, text=f"Downloading... {pct:.0%}")
        elif d['status'] == 'finished':
            progress_placeholder.progress(1.0, text="Converting to WAV...")

    ydl_opts = {
        'format': format_id,
        'outtmpl': os.path.splitext(final_path)[0] + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'progress_hooks': [progress_hook],
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return final_path


def _download_best_audio(url: str, output_path: str, progress_placeholder) -> str:
    """Download best available audio and convert to WAV."""
    import yt_dlp

    final_path = output_path
    if not final_path.endswith('.wav'):
        final_path = os.path.splitext(output_path)[0] + '.wav'

    def progress_hook(d):
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
            downloaded = d.get('downloaded_bytes', 0)
            if total > 0:
                pct = downloaded / total
                progress_placeholder.progress(pct, text=f"Downloading... {pct:.0%}")
        elif d['status'] == 'finished':
            progress_placeholder.progress(1.0, text="Converting to WAV...")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.splitext(final_path)[0] + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'progress_hooks': [progress_hook],
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return final_path


def render_download_audio_tab():
    """Render the Download Audio tab with YouTube audio downloading support."""
    st.subheader("⬇️ Download Audio from YouTube")
    st.caption("Paste a YouTube URL to download audio. Supports UN conference videos with multiple language tracks.")

    # Check yt-dlp availability
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        st.error(
            "**yt-dlp is not installed.** Please install it:\n\n"
            "```bash\npip install yt-dlp\n```"
        )
        return

    # --- URL Input ---
    url = st.text_input(
        "YouTube URL",
        value=st.session_state.get('yt_url', ''),
        placeholder="https://www.youtube.com/watch?v=...",
        key="yt_url_input",
    )

    col_fetch, col_clear = st.columns([1, 1])
    fetch_clicked = col_fetch.button("🔍 Fetch Video Info", disabled=not url, use_container_width=True)
    clear_clicked = col_clear.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        for key in ['yt_url', 'yt_video_info', 'yt_downloaded_files']:
            st.session_state.pop(key, None)
        st.rerun()

    # --- Fetch Info ---
    if fetch_clicked and url:
        st.session_state['yt_url'] = url
        with st.spinner("Fetching video information..."):
            try:
                info = _fetch_video_info(url)
                st.session_state['yt_video_info'] = info
            except Exception as e:
                st.error(f"Failed to fetch video info: {e}")
                logger.exception("yt-dlp fetch info failed")
                return

    # --- Display Info & Download ---
    info = st.session_state.get('yt_video_info')
    if not info:
        st.info("Enter a YouTube URL above and click **Fetch Video Info** to get started.")
        return

    # Video metadata
    st.markdown("---")
    col_thumb, col_meta = st.columns([1, 2])
    with col_thumb:
        thumbnail = info.get('thumbnail')
        if thumbnail:
            st.image(thumbnail, use_container_width=True)

    with col_meta:
        st.markdown(f"**{info.get('title', 'Unknown Title')}**")
        st.caption(f"Channel: {info.get('uploader', 'Unknown')}")
        st.caption(f"Duration: {_format_duration(info.get('duration'))}")
        if info.get('upload_date'):
            date_str = info['upload_date']
            st.caption(f"Upload date: {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")

    # Classify audio tracks
    formats = info.get('formats', [])
    classification = _classify_audio_formats(formats)

    st.markdown("---")

    if classification['is_multi_track']:
        _render_multi_track_download(info, classification)
    else:
        _render_single_track_download(info, classification)

    # --- Show previously downloaded files ---
    downloaded = st.session_state.get('yt_downloaded_files', [])
    if downloaded:
        st.markdown("---")
        st.markdown("### 📁 Downloaded Files")
        for fpath in downloaded:
            if os.path.exists(fpath):
                st.success(f"✅ `{os.path.basename(fpath)}`")
                st.audio(fpath, format='audio/wav')
                st.caption(f"Path: `{os.path.abspath(fpath)}`")
            else:
                st.warning(f"File not found: {fpath}")
        st.info("💡 Go to the **Transcribe Audio** tab to process these files with ASR.")


def _render_multi_track_download(info: dict, classification: dict):
    """Render UI for multi-track (multi-language) audio download."""
    st.markdown("### 🌐 Multiple Audio Tracks Detected")
    st.info("This video has multiple language audio tracks (common in UN conference videos). "
            "Select the tracks you want to download.")

    tracks = classification['tracks']
    lang_options = []
    for lang, fmts in tracks.items():
        best = fmts[0]
        abr = best.get('abr', '?')
        codec = best.get('acodec', '?')
        label = f"{_get_language_label(lang)} — {abr}kbps ({codec})"
        lang_options.append((lang, label, best))

    # Display available tracks
    st.markdown("**Available tracks:**")
    for lang, label, _ in lang_options:
        st.markdown(f"- {label}")

    st.markdown("")

    # Track selection
    col1, col2 = st.columns(2)
    lang_keys = [lo[0] for lo in lang_options]
    lang_labels = [lo[1] for lo in lang_options]

    with col1:
        en_idx = next((i for i, k in enumerate(lang_keys) if k.startswith('en')), 0)
        selected_track_1 = st.selectbox(
            "Track 1 (e.g., Source/English)",
            options=range(len(lang_labels)),
            format_func=lambda i: lang_labels[i],
            index=en_idx,
            key="yt_track1",
        )

    with col2:
        zh_idx = next((i for i, k in enumerate(lang_keys) if k.startswith('zh')), min(1, len(lang_keys) - 1))
        selected_track_2 = st.selectbox(
            "Track 2 (e.g., Interpretation/Chinese)",
            options=range(len(lang_labels)),
            format_func=lambda i: lang_labels[i],
            index=zh_idx,
            key="yt_track2",
        )

    if st.button("⬇️ Download Selected Tracks", use_container_width=True):
        _ensure_download_dir()
        safe_name = _sanitize_filename(info.get('title', 'video'))
        active_url = st.session_state.get('yt_url', '')

        downloaded_files = []
        for idx, track_idx in enumerate([selected_track_1, selected_track_2]):
            lang_code, _, best_fmt = lang_options[track_idx]
            out_path = os.path.join(DOWNLOADS_DIR, f"{safe_name}_{lang_code}.wav")
            progress = st.empty()
            st.markdown(f"**Downloading track: {_get_language_label(lang_code)}...**")
            try:
                result_path = _download_audio(active_url, best_fmt['format_id'], out_path, progress)
                downloaded_files.append(result_path)
                progress.progress(1.0, text=f"✅ {_get_language_label(lang_code)} complete")
            except Exception as e:
                progress.empty()
                st.error(f"Failed to download {_get_language_label(lang_code)} track: {e}")
                logger.exception("yt-dlp multi-track download failed")

        if downloaded_files:
            st.session_state['yt_downloaded_files'] = downloaded_files
            st.rerun()


def _render_single_track_download(info: dict, classification: dict):
    """Render UI for single-track audio download."""
    st.markdown("### 🎵 Single Audio Stream")
    best = classification.get('best_audio')
    if best:
        abr = best.get('abr', '?')
        codec = best.get('acodec', '?')
        st.caption(f"Best audio: {abr}kbps ({codec})")

    st.info("This video has a single audio stream. For stereo recordings (e.g., source on left, "
            "interpretation on right), use channel separation in the **Transcribe Audio** tab after downloading.")

    if st.button("⬇️ Download Audio", use_container_width=True):
        _ensure_download_dir()
        safe_name = _sanitize_filename(info.get('title', 'video'))
        out_path = os.path.join(DOWNLOADS_DIR, f"{safe_name}.wav")
        active_url = st.session_state.get('yt_url', '')

        progress = st.empty()
        try:
            result_path = _download_best_audio(active_url, out_path, progress)
            progress.progress(1.0, text="✅ Download complete")
            st.session_state['yt_downloaded_files'] = [result_path]
            st.rerun()
        except Exception as e:
            progress.empty()
            st.error(f"Download failed: {e}")
            logger.exception("yt-dlp single-track download failed")
