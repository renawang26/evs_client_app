import os
import json
import base64
import tempfile
import subprocess
import pandas as pd
import time
import logging
import warnings
import torch
import streamlit as st
from datetime import datetime
from pathlib import Path
from pydub import AudioSegment
from hashlib import md5
import shutil
import math
import io
import threading
from db_utils import EVSDataUtils


EVS_RESOURCES_PATH = "./evs_resources"

logger = logging.getLogger(__name__)

# FunASR availability flag
_FUNASR_AVAILABLE = None
# CrisperWhisper availability flag
_CRISPERWHISPER_AVAILABLE = None

# Model cache for performance optimization
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()


def get_gpu_info():
    """
    Get GPU availability and info.

    Returns:
        dict: {
            'available': bool,
            'device': str ('cuda' or 'cpu'),
            'name': str (GPU name or 'CPU'),
            'memory_total': float (GB, None for CPU),
            'memory_free': float (GB, None for CPU)
        }
    """
    info = {
        'available': False,
        'device': 'cpu',
        'name': 'CPU',
        'memory_total': None,
        'memory_free': None
    }

    if torch.cuda.is_available():
        info['available'] = True
        info['device'] = 'cuda'
        info['name'] = torch.cuda.get_device_name(0)

        # Get memory info
        try:
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_free = (torch.cuda.get_device_properties(0).total_memory -
                          torch.cuda.memory_allocated(0)) / (1024**3)
            info['memory_total'] = round(memory_total, 2)
            info['memory_free'] = round(memory_free, 2)
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")

    return info


def clear_model_cache():
    """Clear the model cache to free memory."""
    global _MODEL_CACHE
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()
        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")


def get_cache_status():
    """
    Get current model cache status.

    Returns:
        dict: {
            'cached_models': list of model names,
            'count': number of cached models,
            'gpu_memory_used': GPU memory in GB (if available)
        }
    """
    with _MODEL_CACHE_LOCK:
        cached = list(_MODEL_CACHE.keys())

    status = {
        'cached_models': cached,
        'count': len(cached),
        'gpu_memory_used': None
    }

    if torch.cuda.is_available():
        try:
            memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            status['gpu_memory_used'] = round(memory_used, 2)
        except Exception:
            pass

    return status


def get_cached_funasr_model(model_name: str):
    """
    Get a cached FunASR model, loading it only if not already cached.

    Args:
        model_name: Name of the FunASR model

    Returns:
        Loaded FunASR model
    """
    cache_key = f"funasr_{model_name}"

    with _MODEL_CACHE_LOCK:
        if cache_key not in _MODEL_CACHE:
            from funasr import AutoModel

            # Map model names to FunASR model identifiers
            model_mapping = {
                "paraformer-zh": "paraformer-zh",
                "paraformer-en": "paraformer-en",
                "SenseVoiceSmall": "iic/SenseVoiceSmall",
                "sensevoice-small": "iic/SenseVoiceSmall",
            }

            funasr_model_id = model_mapping.get(model_name, model_name)

            logger.info(f"Loading FunASR model '{funasr_model_id}' (first time, will be cached)")
            model = AutoModel(
                model=funasr_model_id,
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                spk_model=None,
            )
            _MODEL_CACHE[cache_key] = model
            logger.info(f"FunASR model '{model_name}' cached successfully")
        else:
            logger.info(f"Using cached FunASR model '{model_name}'")

        return _MODEL_CACHE[cache_key]

def check_funasr_available():
    """Check if FunASR is installed and available"""
    global _FUNASR_AVAILABLE
    if _FUNASR_AVAILABLE is None:
        try:
            from funasr import AutoModel
            _FUNASR_AVAILABLE = True
        except ImportError:
            _FUNASR_AVAILABLE = False
    return _FUNASR_AVAILABLE


def check_crisperwhisper_available():
    """Check if CrisperWhisper (HuggingFace transformers) is installed and available"""
    global _CRISPERWHISPER_AVAILABLE
    if _CRISPERWHISPER_AVAILABLE is None:
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            _CRISPERWHISPER_AVAILABLE = True
        except ImportError:
            _CRISPERWHISPER_AVAILABLE = False
    return _CRISPERWHISPER_AVAILABLE


def get_cached_crisperwhisper_model():
    """
    Get a cached CrisperWhisper pipeline, loading it only if not already cached.

    CrisperWhisper is a fine-tuned Whisper model from nyrahealth that provides
    verbatim English transcription with precise word-level timestamps and filler detection.

    Shows download progress in GUI on first use (~3GB model download).

    Returns:
        HuggingFace ASR pipeline for CrisperWhisper
    """
    cache_key = "crisperwhisper"

    with _MODEL_CACHE_LOCK:
        if cache_key not in _MODEL_CACHE:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            from huggingface_hub import snapshot_download, HfApi
            import os

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            model_id = "nyrahealth/CrisperWhisper"

            logger.info(f"Loading CrisperWhisper model on {device} (first time, will be cached)")

            # Check if model weights are already downloaded locally
            model_already_downloaded = False
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(model_id, "model.safetensors", local_files_only=True)
                model_already_downloaded = True
            except Exception:
                model_already_downloaded = False

            status_container = st.empty()

            if not model_already_downloaded:
                # First-time download — show status in GUI
                status_container.info(
                    "⏳ **Downloading CrisperWhisper model (~3GB)** — this is a one-time download. "
                    "Check the console/terminal for download progress. "
                    "Future runs will use the cached model."
                )

                # Download all model files first (progress shown in console via tqdm)
                try:
                    snapshot_download(model_id, local_files_only=False)
                except Exception as dl_err:
                    logger.warning(f"snapshot_download warning: {dl_err}")

            # Load model from local cache
            status_container.info("📦 **Loading CrisperWhisper model into memory...** this may take a minute.")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=model_already_downloaded,
            )

            status_container.info("📦 **Moving model to GPU...**")
            model.to(device)

            processor = AutoProcessor.from_pretrained(model_id)

            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=dtype,
                device=device,
                return_timestamps="word",
            )

            _MODEL_CACHE[cache_key] = pipe
            logger.info("CrisperWhisper model cached successfully")

            status_container.success("CrisperWhisper model loaded successfully!")
            import time as _time
            _time.sleep(2)
            status_container.empty()
        else:
            logger.info("Using cached CrisperWhisper model")

        return _MODEL_CACHE[cache_key]

class ASRUtils:

    # Last transcription error message (for UI display)
    _last_error = None

    # 添加线程锁用于Google转录
    google_transcribe_lock = threading.Lock()

    @staticmethod
    def process_audio_file_with_ibm(file_path, target_path, api_key, language):
        """
        Process audio file with IBM Watson
        """
        try:
            # Import IBM Watson related libraries
            from ibm_watson import SpeechToTextV1
            from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

            # Set up IBM Watson API credentials
            authenticator = IAMAuthenticator(api_key)
            speech_to_text = SpeechToTextV1(authenticator=authenticator)
            speech_to_text.set_service_url('https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/your-instance-id')

            # Transcribe audio
            result = ASRUtils.transcribe_with_ibm(file_path, speech_to_text, language)

            # Save results
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

            return result
        except Exception as e:
            logger.error(f"Error processing audio with IBM: {str(e)}")
            return None

    @staticmethod
    def get_masked_filename(base_filename):
        if base_filename is None:
            return None

        file_extension = os.path.splitext(base_filename)[1]

        # 使用MD5生成文件名哈希并截取前8位作为掩码
        name_hash = md5(base_filename.encode()).hexdigest()[:8]
        masked_filename = f"file_{name_hash}{file_extension}"

        return masked_filename

    @staticmethod
    def transcribe_with_tencent(file_path, secret_id, secret_key, asr_provider, tencent_model, language, base_filename, duration, audio_file, channel_num):
        try:
            logger.info(f"Using alternative method to process audio: {audio_file}, language: {language}, channel_num: {channel_num}")

            # Read audio file
            audio_content = AudioSegment.from_file(file_path)

            # Tencent Cloud ASR maximum request size (10MB)
            MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB in bytes

            # Create temporary directory for audio chunks
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory for audio chunks: {temp_dir}")

            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info("Preparing audio for transcription...")

            # Get audio file size
            with open(file_path, 'rb') as f:
                f.seek(0, 2)  # Go to the end of the file
                file_size = f.tell()  # Get current position (file size)

            # Determine if splitting is needed
            if file_size <= MAX_REQUEST_SIZE:
                # File is small enough, process directly
                with open(file_path, 'rb') as f:
                    audio_content = f.read()

                # Process the single chunk
                audio_base64 = base64.b64encode(audio_content).decode('utf-8')
                words_df, segments_df = ASRUtils._process_tencent_chunk(
                    audio_base64, secret_id, secret_key, language, base_filename,
                    asr_provider, tencent_model, duration, audio_file, channel_num, 0, 1
                )

                # Clean up
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.error(f"Error removing temp directory: {e}")

                return words_df, segments_df

            # Calculate how many chunks we need
            chunk_size_ms = 60000  # 1 minute chunks in milliseconds
            total_duration_ms = len(audio_content)
            chunks_count = math.ceil(total_duration_ms / chunk_size_ms)

            logger.info(f"Splitting audio file into {chunks_count} chunks")
            status_text.info(f"Splitting audio into {chunks_count} chunks for processing...")

            # Split audio into chunks and process each one
            all_words_data = []
            all_segments_data = []

            for i in range(chunks_count):
                # Update progress
                progress_bar.progress(i / chunks_count)
                status_text.info(f"Processing chunk {i+1} of {chunks_count}...")

                # Calculate chunk start and end times
                start_ms = i * chunk_size_ms
                end_ms = min((i + 1) * chunk_size_ms, total_duration_ms)
                chunk_start_second = start_ms / 1000.0
                chunk_end_second = end_ms / 1000.0

                # Extract the chunk
                chunk = audio_content[start_ms:end_ms]

                # Save chunk to temp file
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                chunk.export(chunk_path, format="wav")

                # Read chunk data
                with open(chunk_path, 'rb') as f:
                    chunk_data = f.read()

                # Convert to base64
                chunk_base64 = base64.b64encode(chunk_data).decode('utf-8')

                # Process chunk
                try:
                    words_df, segments_df = ASRUtils._process_tencent_chunk(
                        chunk_base64, secret_id, secret_key, language, base_filename,
                        asr_provider, tencent_model, duration, audio_file, channel_num,
                        chunk_start_second, chunks_count, chunk_index=i
                    )

                    # 不需要再手动调整时间戳，因为这已经在_process_tencent_chunk中处理了
                    if not words_df.empty:
                        all_words_data.append(words_df)

                    if not segments_df.empty:
                        all_segments_data.append(segments_df)

                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    status_text.warning(f"Warning: Error processing chunk {i+1}. Will continue with other chunks.")

            # Clean up temporary files
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error removing temp directory: {e}")

            # Combine all results
            if all_words_data:
                combined_words_df = pd.concat(all_words_data, ignore_index=True)
            else:
                combined_words_df = pd.DataFrame()

            if all_segments_data:
                combined_segments_df = pd.concat(all_segments_data, ignore_index=True)
            else:
                combined_segments_df = pd.DataFrame()

            # Complete progress
            progress_bar.progress(100)
            status_text.success("Audio processing completed!")

            return combined_words_df, combined_segments_df

        except Exception as e:
            logger.error(f"Error processing audio file: {e}", exc_info=True)
            st.error(f"Error processing audio file: {e}")
            return pd.DataFrame(), pd.DataFrame()

    @staticmethod
    def _process_tencent_chunk(audio_base64, secret_id, secret_key, language, base_filename,
                               asr_provider, tencent_model, duration, audio_file, channel_num,
                               time_offset=0.0, total_chunks=1, chunk_index=0):
        """Process a single audio chunk and transcribe it"""
        try:
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.asr.v20190614 import asr_client, models
            from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException

            # Choose engine model type based on language
            engine_model_type = "16k_zh" if language == "zh" else "16k_en"

            # Current chunk start and end time
            chunk_start_time = time_offset
            if chunk_index < total_chunks - 1:
                chunk_end_time = time_offset + 60.0  # Assume each non-last chunk is 60 seconds
            else:
                # For the last chunk, we can set an estimated value or a special marker
                chunk_end_time = time_offset + 60.0  # Assume it's also 60 seconds, or use actual audio length
                # Or keep it as a special marker value, but not None

            # Initialize Tencent Cloud API client
            cred = credential.Credential(secret_id, secret_key)
            httpProfile = HttpProfile()
            httpProfile.endpoint = "asr.tencentcloudapi.com"

            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            client = asr_client.AsrClient(cred, "ap-guangzhou", clientProfile)

            # Create request
            req = models.CreateRecTaskRequest()
            params = {
                "EngineModelType": engine_model_type,
                "ChannelNum": 1,
                "ResTextFormat": 2,
                "SourceType": 1,
                "Data": audio_base64
            }
            req.from_json_string(json.dumps(params))

            # Send request
            with st.spinner(f"Submitting transcription request for chunk {chunk_index+1}/{total_chunks}..."):
                resp = client.CreateRecTask(req)
                resp_json = resp.to_json_string()
                logger.info(f"Task creation response: {resp_json}")

            # Parse response to get TaskId
            resp_dict = json.loads(resp_json)
            task_id = resp_dict.get("Data", {}).get("TaskId")

            if not task_id:
                logger.error("Failed to get TaskId")
                st.error("Failed to get TaskId, please check API response")
                st.json(resp_dict)
                return pd.DataFrame(), pd.DataFrame()

            st.success(f"Successfully created transcription task for chunk {chunk_index+1}/{total_chunks}, Task ID: {task_id}")

            # Poll for results
            max_retries = 10  # Maximum polling attempts
            retry_delay = 2   # Polling interval (seconds)

            # Create status indicator
            status_text = st.empty()

            # Wait and get results
            result_data = None
            for retry in range(max_retries):

                status_text.info(f"Checking transcription status for chunk {chunk_index+1}/{total_chunks}... (Attempt {retry+1}/{max_retries})")

                # Query task status
                query_req = models.DescribeTaskStatusRequest()
                query_params = {"TaskId": task_id}
                query_req.from_json_string(json.dumps(query_params))

                try:
                    query_resp = client.DescribeTaskStatus(query_req)
                    query_resp_json = query_resp.to_json_string()
                    logger.info(f"Task status response: {query_resp_json}")

                    query_resp_dict = json.loads(query_resp_json)
                    status_code = query_resp_dict.get("Data", {}).get("Status")

                    # Update status text
                    status_text_map = {0: "Waiting", 1: "Processing", 2: "Success", 3: "Failed"}
                    status_label = status_text_map.get(status_code, f"Unknown ({status_code})")

                    st.info(f"Status check {retry+1}: {status_label} at {datetime.now().isoformat()}\n")

                    # Process based on status
                    if status_code == 2:  # Success
                        # Save original response JSON for reference
                        log_dir = './logs'
                        os.makedirs(log_dir, exist_ok=True)
                        result_file = os.path.join(log_dir, f"{base_filename}_{language}_tencent_chunk{chunk_index}.json")
                        with open(result_file, 'w', encoding='utf-8') as f:
                            f.write(query_resp_json)

                        status_text.success(f"Transcription completed successfully for chunk {chunk_index+1}/{total_chunks}!")

                        # Use query response as result data
                        result_data = query_resp_dict.get("Data", {})

                        words_data = []
                        segments_data = []

                        # Process transcription results
                        for segment_id, sentence in enumerate(result_data.get("ResultDetail", [])):
                            # Adjust timestamps to account for chunk offset
                            segment_start = sentence.get("StartMs", 0) / 1000.0 + time_offset
                            segment_end = sentence.get("EndMs", 0) / 1000.0 + time_offset
                            segment_text = sentence.get("FinalSentence", "")
                            segment_duration = segment_end - segment_start

                            # Add segment data, including chunk information
                            segments_data.append({
                                'file_name': base_filename,
                                'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                                'lang': language,
                                'segment_id': segment_id + (chunk_index * 1000),  # Ensure segment IDs from different chunks don't conflict
                                'start_time': segment_start,
                                'end_time': segment_end,
                                'text': segment_text,
                                'speaker': None,
                                'duration': segment_duration,
                                'slice_duration': duration,
                                'asr_provider': asr_provider,
                                'model': tencent_model if isinstance(tencent_model, str) else str(tencent_model).split('(')[0],
                                'channel_num': channel_num,
                                'audio_file': audio_file,
                                'chunk_start_time': chunk_start_time,  # Add chunk start time
                                'chunk_end_time': chunk_end_time,      # Add chunk end time
                                'chunk_id': chunk_index                # Keep unique chunk ID
                            })

                            # Process word-level data
                            for word_seq_no, word_info in enumerate(sentence.get("Words", [])):
                                word_text = word_info.get("Word", "")
                                word_start = (sentence.get("StartMs", 0) + word_info.get("OffsetStartMs", 0)) / 1000.0 + time_offset
                                word_end = (sentence.get("StartMs", 0) + word_info.get("OffsetEndMs", 0)) / 1000.0 + time_offset
                                word_duration = word_end - word_start

                                words_data.append({
                                    'file_name': base_filename,
                                    'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                                    'lang': language,
                                    'word': word_text,
                                    'start_time': word_start,
                                    'end_time': word_end,
                                    'confidence': word_info.get("WordConfidence", 1.0) / 100,  # Convert to 0-1 range
                                    'speaker': None,
                                    'segment_id': segment_id + (chunk_index * 1000),  # Ensure it matches segment ID
                                    'word_seq_no': word_seq_no,
                                    'duration': word_duration,
                                    'slice_duration': duration,
                                    'asr_provider': asr_provider,
                                    'model': tencent_model if isinstance(tencent_model, str) else str(tencent_model).split('(')[0],
                                    'channel_num': channel_num,
                                    'audio_file': audio_file,
                                    'chunk_start_time': chunk_start_time,  # Add chunk start time
                                    'chunk_end_time': chunk_end_time,      # Add chunk end time
                                    'chunk_id': chunk_index                # Keep unique chunk ID
                                })

                        # Create DataFrames
                        words_df = pd.DataFrame(words_data) if words_data else pd.DataFrame()
                        segments_df = pd.DataFrame(segments_data) if segments_data else pd.DataFrame()

                        return words_df, segments_df

                    elif status_code == 3:  # Failed
                        error_msg = query_resp_dict.get("Data", {}).get("ErrorMsg", "Unknown error")
                        logger.error(f"Transcription failed: {error_msg}")
                        status_text.error(f"Transcription failed for chunk {chunk_index+1}: {error_msg}")
                        return pd.DataFrame(), pd.DataFrame()

                    # For status 0 or 1, wait and continue polling
                    time.sleep(retry_delay)

                except Exception as e:
                    logger.error(f"Error during polling: {e}")
                    status_text.error.warning(f"Error checking status: {str(e)}")
                    time.sleep(retry_delay)

            # Process polling end without success
            if result_data is None:
                st.error(f"Maximum polling attempts reached for chunk {chunk_index+1}. The task may still be processing.")
                status_text.info(f"Maximum polling attempts reached for chunk {chunk_index+1}. The task may still be processing.")

                return pd.DataFrame(), pd.DataFrame()

        except TencentCloudSDKException as e:
            logger.error(f"Tencent Cloud SDK exception: {e}")
            st.error(f"Tencent Cloud SDK exception: {e}")
            return pd.DataFrame(), pd.DataFrame()

    @staticmethod
    def _get_language_detect_model():
        """Get a cached whisper-tiny model for language detection (~39MB)."""
        cache_key = "whisper_tiny_langdetect"
        if cache_key not in _MODEL_CACHE:
            from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
            import torch

            logger.info("Loading whisper-tiny for language detection (~39MB)...")
            model_id = "openai/whisper-tiny"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
            feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
            tokenizer = WhisperTokenizer.from_pretrained(model_id)

            _MODEL_CACHE[cache_key] = {
                'model': model, 'feature_extractor': feature_extractor,
                'tokenizer': tokenizer, 'device': device
            }
            logger.info("whisper-tiny loaded for language detection")

        return _MODEL_CACHE[cache_key]

    @staticmethod
    def detect_language(file_path, model_name="tiny"):
        """
        Detect language of audio file using whisper-tiny's built-in language detection.

        Uses openai/whisper-tiny (~39MB) which has calibrated multilingual
        language detection, unlike CrisperWhisper which is English-only.

        Args:
            file_path: Path to audio file
            model_name: Unused (kept for backward compatibility)

        Returns:
            tuple: (language_code, confidence) e.g., ("en", 0.95) or ("zh", 0.87)
                   Returns (None, 0.0) if detection fails
        """
        try:
            # Convert Path object to string
            if hasattr(file_path, 'resolve'):
                file_path = str(file_path)

            logger.info(f"Detecting language for: {file_path}")

            if not check_crisperwhisper_available():
                logger.warning("transformers not available for language detection")
                return None, 0.0

            import torchaudio
            import torch

            # Load and prepare audio: mono 16kHz, first 30 seconds
            waveform, sample_rate = torchaudio.load(file_path)
            logger.info(f"Language detect audio: shape={waveform.shape}, sr={sample_rate}")
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            max_samples = sample_rate * 30
            waveform = waveform[:, :max_samples]
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            audio_np = waveform.squeeze().numpy()

            if len(audio_np) < 1600:
                logger.warning("Audio too short for language detection")
                return None, 0.0

            # Use whisper-tiny for language detection (properly calibrated)
            lang_model = ASRUtils._get_language_detect_model()
            model = lang_model['model']
            feature_extractor = lang_model['feature_extractor']
            tokenizer = lang_model['tokenizer']
            device = lang_model['device']

            # Extract mel features
            input_features = feature_extractor(
                audio_np, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)

            # Use Whisper's language detection: predict language token
            with torch.no_grad():
                decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device)
                outputs = model(input_features, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits[0, 0]

                # Get language token probabilities
                probs = torch.softmax(logits, dim=-1)

                en_token_id = tokenizer.convert_tokens_to_ids("<|en|>")
                zh_token_id = tokenizer.convert_tokens_to_ids("<|zh|>")

                en_prob = probs[en_token_id].item() if en_token_id is not None else 0.0
                zh_prob = probs[zh_token_id].item() if zh_token_id is not None else 0.0

                logger.info(f"Language probabilities - en: {en_prob:.4f}, zh: {zh_prob:.4f}")

                if en_prob > zh_prob:
                    detected_lang = "en"
                    confidence = en_prob / (en_prob + zh_prob) if (en_prob + zh_prob) > 0 else 0.5
                else:
                    detected_lang = "zh"
                    confidence = zh_prob / (en_prob + zh_prob) if (en_prob + zh_prob) > 0 else 0.5

            logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.2%})")
            return detected_lang, confidence

        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}", exc_info=True)
            st.warning(f"Language detection error: {str(e)}")
            return None, 0.0

    @staticmethod
    def detect_channel_languages(channel1_path, channel2_path, model_name="tiny"):
        """
        Detect languages for both audio channels and determine if swap is needed.

        Args:
            channel1_path: Path to channel 1 audio file
            channel2_path: Path to channel 2 audio file
            model_name: Whisper model to use for detection

        Returns:
            dict: {
                'channel1': {'language': 'en', 'confidence': 0.95},
                'channel2': {'language': 'zh', 'confidence': 0.87},
                'swap_recommended': False,  # True if ch1=zh and ch2=en
                'detection_successful': True
            }
        """
        result = {
            'channel1': {'language': None, 'confidence': 0.0},
            'channel2': {'language': None, 'confidence': 0.0},
            'swap_recommended': False,
            'detection_successful': False
        }

        try:
            # Detect language for channel 1
            lang1, conf1 = ASRUtils.detect_language(channel1_path, model_name)
            result['channel1'] = {'language': lang1, 'confidence': conf1}

            # Detect language for channel 2
            lang2, conf2 = ASRUtils.detect_language(channel2_path, model_name)
            result['channel2'] = {'language': lang2, 'confidence': conf2}

            # If only one channel detected, infer the other
            # (CrisperWhisper is English-only, so Chinese channel may return None)
            if lang1 and not lang2:
                lang2 = 'zh' if lang1 == 'en' else 'en'
                conf2 = 0.7  # inferred confidence
                result['channel2'] = {'language': lang2, 'confidence': conf2}
                logger.info(f"Inferred channel 2 language: {lang2} (from channel 1={lang1})")
            elif lang2 and not lang1:
                lang1 = 'zh' if lang2 == 'en' else 'en'
                conf1 = 0.7  # inferred confidence
                result['channel1'] = {'language': lang1, 'confidence': conf1}
                logger.info(f"Inferred channel 1 language: {lang1} (from channel 2={lang2})")

            # Check if detection was successful
            if lang1 and lang2:
                result['detection_successful'] = True

                # Recommend swap if channel 1 is Chinese and channel 2 is English
                # (Default assumption is ch1=English, ch2=Chinese)
                if lang1 == 'zh' and lang2 == 'en':
                    result['swap_recommended'] = True
                elif lang1 == 'en' and lang2 == 'zh':
                    result['swap_recommended'] = False

            logger.info(f"Channel language detection complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Channel language detection failed: {str(e)}", exc_info=True)
            return result

    @staticmethod
    def transcribe_with_funasr(file_path, asr_provider, model_name, language, base_filename, duration, audio_file, channel_num):
        """
        Transcribe audio using FunASR (Alibaba's ASR framework)
        Excellent for Chinese speech recognition using Paraformer model

        Args:
            file_path: Audio file path
            asr_provider: ASR provider name (e.g. 'funasr')
            model_name: FunASR model name (e.g. 'paraformer-zh', 'SenseVoiceSmall')
            language: Language code (e.g. 'zh', 'en')
            base_filename: Base file name (for database)
            duration: Audio duration
            audio_file: Audio file path (for database)
            channel_num: Audio channel number

        Returns:
            tuple: (words_df, segments_df) - Word and segment dataframes
        """
        try:
            if not check_funasr_available():
                err = "FunASR is not installed. Install with: pip install funasr modelscope torch torchaudio"
                logger.error(err)
                ASRUtils._last_error = err
                import pandas as pd
                return pd.DataFrame(), pd.DataFrame()

            # Convert Path object to string
            if hasattr(file_path, 'resolve'):
                file_path = str(file_path)

            # Validate file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            # Get cached FunASR model
            model = get_cached_funasr_model(model_name)

            # Transcribe audio
            logger.info(f"Transcribing with FunASR: {file_path}")
            result = model.generate(
                input=file_path,
                batch_size_s=300,  # Process 300 seconds at a time
                hotword=None,
            )

            # Save transcription result to file
            output_prefix = os.path.splitext(base_filename)[0]
            result_dir = os.path.join(EVS_RESOURCES_PATH, "slice_audio_files", output_prefix, asr_provider)
            os.makedirs(result_dir, exist_ok=True)
            result_file = os.path.join(result_dir, f"{output_prefix}_{model_name}_{language}_{asr_provider}.json")

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4, default=str)

            # Prepare database data
            words_data = []
            segments_data = []

            # Process FunASR results
            # FunASR returns list of results, one per audio file
            for res in result:
                text = res.get("text", "")
                timestamp = res.get("timestamp", [])  # Character-level timestamps for Chinese

                # If timestamp data available, extract word-level timing
                if timestamp and len(timestamp) > 0:
                    # For Chinese: timestamps are character-level, use jieba for word segmentation
                    if language == 'zh':
                        # Import jieba for Chinese word segmentation
                        import jieba

                        # Get characters from text (excluding spaces)
                        chars = [c for c in text if c.strip()]

                        # Segment text into words using jieba
                        words_list = list(jieba.cut(text))

                        # Build character to timestamp mapping
                        char_timestamps = []
                        ts_idx = 0
                        for char in text:
                            if char.strip() and ts_idx < len(timestamp):
                                ts = timestamp[ts_idx]
                                if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                                    char_timestamps.append({
                                        'char': char,
                                        'start': ts[0] / 1000.0,
                                        'end': ts[1] / 1000.0
                                    })
                                ts_idx += 1

                        # Map words to timestamps by accumulating character positions
                        word_seq_no = 0
                        char_pos = 0
                        for word_text in words_list:
                            word_text_clean = word_text.strip()
                            if not word_text_clean:
                                continue

                            # Find character timestamps for this word
                            word_char_count = len(word_text_clean)
                            word_char_timestamps = []

                            # Match characters in word to their timestamps
                            temp_pos = char_pos
                            for wc in word_text_clean:
                                if temp_pos < len(char_timestamps):
                                    word_char_timestamps.append(char_timestamps[temp_pos])
                                    temp_pos += 1

                            if word_char_timestamps:
                                word_start = word_char_timestamps[0]['start']
                                word_end = word_char_timestamps[-1]['end']
                                char_pos = temp_pos
                            else:
                                # Fallback if no timestamps found
                                word_start = word_seq_no * 0.5
                                word_end = word_start + 0.5

                            words_data.append({
                                'file_name': base_filename,
                                'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                                'lang': language,
                                'word': word_text_clean,
                                'start_time': word_start,
                                'end_time': word_end,
                                'confidence': 1.0,
                                'speaker': None,
                                'segment_id': 0,
                                'word_seq_no': word_seq_no,
                                'duration': word_end - word_start,
                                'slice_duration': duration,
                                'asr_provider': asr_provider,
                                'model': model_name,
                                'channel_num': channel_num,
                                'audio_file': audio_file
                            })
                            word_seq_no += 1
                    else:
                        # For non-Chinese: timestamps are word-level, split by whitespace
                        words_from_text = text.split()

                        for word_seq_no, (word_text, ts) in enumerate(zip(words_from_text, timestamp)):
                            if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                                word_start = ts[0] / 1000.0
                                word_end = ts[1] / 1000.0
                            else:
                                word_start = word_seq_no * 0.3
                                word_end = word_start + 0.3

                            words_data.append({
                                'file_name': base_filename,
                                'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                                'lang': language,
                                'word': word_text,
                                'start_time': word_start,
                                'end_time': word_end,
                                'confidence': 1.0,
                                'speaker': None,
                                'segment_id': 0,
                                'word_seq_no': word_seq_no,
                                'duration': word_end - word_start,
                                'slice_duration': duration,
                                'asr_provider': asr_provider,
                                'model': model_name,
                                'channel_num': channel_num,
                                'audio_file': audio_file
                            })

                    # Create segment from full text
                    if words_data:
                        segments_data.append({
                            'file_name': base_filename,
                            'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                            'lang': language,
                            'segment_id': 0,
                            'start_time': words_data[0]['start_time'],
                            'end_time': words_data[-1]['end_time'],
                            'text': text,
                            'speaker': None,
                            'duration': words_data[-1]['end_time'] - words_data[0]['start_time'],
                            'slice_duration': duration,
                            'asr_provider': asr_provider,
                            'model': model_name,
                            'channel_num': channel_num,
                            'audio_file': audio_file
                        })
                else:
                    # No timestamps available, create segment only
                    segments_data.append({
                        'file_name': base_filename,
                        'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                        'lang': language,
                        'segment_id': 0,
                        'start_time': 0,
                        'end_time': duration,
                        'text': text,
                        'speaker': None,
                        'duration': duration,
                        'slice_duration': duration,
                        'asr_provider': asr_provider,
                        'model': model_name,
                        'channel_num': channel_num,
                        'audio_file': audio_file
                    })

                    # Create word entries by splitting text
                    for word_seq_no, word_text in enumerate(text.split()):
                        estimated_duration = duration / max(len(text.split()), 1)
                        word_start = word_seq_no * estimated_duration
                        word_end = word_start + estimated_duration

                        words_data.append({
                            'file_name': base_filename,
                            'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                            'lang': language,
                            'word': word_text,
                            'start_time': word_start,
                            'end_time': word_end,
                            'confidence': 1.0,
                            'speaker': None,
                            'segment_id': 0,
                            'word_seq_no': word_seq_no,
                            'duration': estimated_duration,
                            'slice_duration': duration,
                            'asr_provider': asr_provider,
                            'model': model_name,
                            'channel_num': channel_num,
                            'audio_file': audio_file
                        })

            # Convert to DataFrame
            import pandas as pd
            words_df = pd.DataFrame(words_data)
            segments_df = pd.DataFrame(segments_data)

            logger.info(f"FunASR transcription complete: {len(words_data)} words, {len(segments_data)} segments")
            return words_df, segments_df

        except Exception as e:
            err = f"Error transcribing with FunASR: {str(e)}"
            logger.error(err)
            import traceback
            logger.error(traceback.format_exc())
            ASRUtils._last_error = err
            import pandas as pd
            return pd.DataFrame(), pd.DataFrame()

    @staticmethod
    def transcribe_with_crisperwhisper(file_path, asr_provider, model_name, language, base_filename, duration, audio_file, channel_num):
        """
        Transcribe audio using CrisperWhisper (nyrahealth/CrisperWhisper)
        Verbatim English ASR with precise word-level timestamps and filler detection.

        Args:
            file_path: Audio file path
            asr_provider: ASR provider name ('crisperwhisper')
            model_name: Model name ('crisperwhisper')
            language: Language code (should be 'en' - CrisperWhisper is English-only)
            base_filename: Base file name (for database)
            duration: Audio duration
            audio_file: Audio file path (for database)
            channel_num: Audio channel number

        Returns:
            tuple: (words_df, segments_df) - Word and segment dataframes
        """
        try:
            if not check_crisperwhisper_available():
                err = "CrisperWhisper requires HuggingFace transformers. Install with: pip install transformers torch torchaudio"
                logger.error(err)
                ASRUtils._last_error = err
                return pd.DataFrame(), pd.DataFrame()

            if language != 'en':
                logger.warning(f"CrisperWhisper is optimized for English only. Language '{language}' may produce poor results.")

            # Convert Path object to string
            if hasattr(file_path, 'resolve'):
                file_path = str(file_path)

            # Validate file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            # Get cached CrisperWhisper pipeline
            pipe = get_cached_crisperwhisper_model()

            # Load audio, ensure mono 16kHz for CrisperWhisper
            logger.info(f"Transcribing with CrisperWhisper: {file_path}")
            import torchaudio
            waveform, sample_rate = torchaudio.load(file_path)
            logger.info(f"Audio loaded: shape={waveform.shape}, sr={sample_rate}")
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            audio_np = waveform.squeeze().numpy()

            # Process audio in manual chunks to avoid pipeline chunking bugs
            audio_duration_s = len(audio_np) / 16000
            logger.info(f"Audio duration: {audio_duration_s:.1f}s")

            # Process audio in manual chunks (25s each, Whisper max is 30s)
            import numpy as np
            chunk_seconds = 25
            chunk_samples = chunk_seconds * 16000
            all_chunks = []
            all_text_parts = []
            offset_samples = 0
            total_chunks = max(1, int(np.ceil(len(audio_np) / chunk_samples)))

            logger.info(f"Audio duration: {audio_duration_s:.1f}s, processing in {total_chunks} chunk(s)")

            while offset_samples < len(audio_np):
                end_samples = min(offset_samples + chunk_samples, len(audio_np))
                chunk = audio_np[offset_samples:end_samples]

                # Skip very short trailing chunks
                if len(chunk) < 8000:  # less than 0.5s
                    break

                offset_seconds = offset_samples / 16000
                chunk_num = offset_samples // chunk_samples + 1

                # Use num_beams=1 to fix tensor mismatch in _extract_token_timestamps
                # CrisperWhisper defaults to num_beams=5 which is incompatible with word timestamps
                chunk_result = pipe(chunk, return_timestamps="word", generate_kwargs={"num_beams": 1})

                # Adjust timestamps by chunk offset
                for word_chunk in chunk_result.get("chunks", []):
                    ts = word_chunk.get("timestamp", (None, None))
                    if ts and ts[0] is not None:
                        word_chunk["timestamp"] = (
                            ts[0] + offset_seconds,
                            ts[1] + offset_seconds if ts[1] is not None else None
                        )
                    all_chunks.append(word_chunk)

                all_text_parts.append(chunk_result.get("text", ""))
                logger.info(f"Chunk {chunk_num}/{total_chunks}: {len(chunk)/16000:.1f}s, {len(chunk_result.get('chunks', []))} words")
                offset_samples = end_samples

            result = {
                "text": " ".join(all_text_parts),
                "chunks": all_chunks
            }

            # Save transcription result to file
            output_prefix = os.path.splitext(base_filename)[0]
            result_dir = os.path.join(EVS_RESOURCES_PATH, "slice_audio_files", output_prefix, asr_provider)
            os.makedirs(result_dir, exist_ok=True)
            result_file = os.path.join(result_dir, f"{output_prefix}_{model_name}_{language}_{asr_provider}.json")

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4, default=str)

            # Prepare database data
            words_data = []
            segments_data = []

            # Process CrisperWhisper results
            # Result format: {"text": "...", "chunks": [{"text": "word", "timestamp": (start, end)}, ...]}
            text = result.get("text", "")
            chunks = result.get("chunks", [])

            if chunks:
                # Build words from chunks (word-level timestamps)
                for word_seq_no, chunk in enumerate(chunks):
                    word_text = chunk.get("text", "").strip()
                    if not word_text:
                        continue

                    ts = chunk.get("timestamp", (None, None))
                    if ts and len(ts) >= 2 and ts[0] is not None and ts[1] is not None:
                        word_start = float(ts[0])
                        word_end = float(ts[1])
                    else:
                        # Fallback: estimate timestamps
                        word_start = word_seq_no * 0.3
                        word_end = word_start + 0.3

                    words_data.append({
                        'file_name': base_filename,
                        'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                        'lang': language,
                        'word': word_text,
                        'start_time': word_start,
                        'end_time': word_end,
                        'confidence': 1.0,
                        'speaker': None,
                        'segment_id': 0,
                        'word_seq_no': word_seq_no,
                        'duration': word_end - word_start,
                        'slice_duration': duration,
                        'asr_provider': asr_provider,
                        'model': model_name,
                        'channel_num': channel_num,
                        'audio_file': audio_file
                    })

                # Create a single segment from all words
                if words_data:
                    segments_data.append({
                        'file_name': base_filename,
                        'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                        'lang': language,
                        'segment_id': 0,
                        'start_time': words_data[0]['start_time'],
                        'end_time': words_data[-1]['end_time'],
                        'text': text,
                        'speaker': None,
                        'duration': words_data[-1]['end_time'] - words_data[0]['start_time'],
                        'slice_duration': duration,
                        'asr_provider': asr_provider,
                        'model': model_name,
                        'channel_num': channel_num,
                        'audio_file': audio_file
                    })
            else:
                # No chunks (no word-level timestamps), create segment only
                segments_data.append({
                    'file_name': base_filename,
                    'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                    'lang': language,
                    'segment_id': 0,
                    'start_time': 0,
                    'end_time': duration,
                    'text': text,
                    'speaker': None,
                    'duration': duration,
                    'slice_duration': duration,
                    'asr_provider': asr_provider,
                    'model': model_name,
                    'channel_num': channel_num,
                    'audio_file': audio_file
                })

                # Create word entries by splitting text
                for word_seq_no, word_text in enumerate(text.split()):
                    estimated_duration = duration / max(len(text.split()), 1)
                    word_start = word_seq_no * estimated_duration
                    word_end = word_start + estimated_duration

                    words_data.append({
                        'file_name': base_filename,
                        'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                        'lang': language,
                        'word': word_text,
                        'start_time': word_start,
                        'end_time': word_end,
                        'confidence': 1.0,
                        'speaker': None,
                        'segment_id': 0,
                        'word_seq_no': word_seq_no,
                        'duration': estimated_duration,
                        'slice_duration': duration,
                        'asr_provider': asr_provider,
                        'model': model_name,
                        'channel_num': channel_num,
                        'audio_file': audio_file
                    })

            # Convert to DataFrame
            words_df = pd.DataFrame(words_data)
            segments_df = pd.DataFrame(segments_data)

            logger.info(f"CrisperWhisper transcription complete: {len(words_data)} words, {len(segments_data)} segments")
            return words_df, segments_df

        except Exception as e:
            err = f"Error transcribing with CrisperWhisper: {str(e)}"
            logger.error(err)
            import traceback
            logger.error(traceback.format_exc())
            ASRUtils._last_error = err
            return pd.DataFrame(), pd.DataFrame()

    @staticmethod
    def transcribe_audio(file_path, provider, model_name, language, base_filename, duration, audio_file, channel_num):
        """
        Universal transcription method that routes to the appropriate ASR provider

        Args:
            file_path: Audio file path
            provider: ASR provider name ('crisperwhisper', 'funasr', 'google', 'tencent')
            model_name: Model name
            language: Language code ('en', 'zh')
            base_filename: Base file name
            duration: Audio duration
            audio_file: Audio file path
            channel_num: Audio channel number

        Returns:
            tuple: (words_df, segments_df)
        """
        ASRUtils._last_error = None
        if provider == "funasr":
            return ASRUtils.transcribe_with_funasr(
                file_path, provider, model_name, language,
                base_filename, duration, audio_file, channel_num
            )
        elif provider == "crisperwhisper":
            return ASRUtils.transcribe_with_crisperwhisper(
                file_path, provider, model_name, language,
                base_filename, duration, audio_file, channel_num
            )
        else:
            # Unknown provider - log warning and fall back to CrisperWhisper for EN, FunASR for ZH
            if language == "zh":
                logger.warning(f"Unknown ASR provider '{provider}', falling back to FunASR for Chinese")
                return ASRUtils.transcribe_with_funasr(
                    file_path, "funasr", "paraformer-zh", language,
                    base_filename, duration, audio_file, channel_num
                )
            else:
                logger.warning(f"Unknown ASR provider '{provider}', falling back to CrisperWhisper for English")
                return ASRUtils.transcribe_with_crisperwhisper(
                    file_path, "crisperwhisper", "crisperwhisper", language,
                    base_filename, duration, audio_file, channel_num
                )

    @staticmethod
    def transcribe_with_google(file_path, credentials_file, base_filename, language, model_name, duration, audio_file, channel_num):
        """
        Transcribe audio using Google Speech-to-Text

        Args:
            file_path: Audio file path
            credentials_file: Path to Google credentials JSON file
            base_filename: Base file name (for database)
            language: Language code (e.g. 'zh-CN', 'en-US')
            model_name: Google model name
            duration: Audio duration
            audio_file: Audio file path (for database)
            channel_num: Audio channel number

        Returns:
            tuple: (words_df, segments_df) - Word and segment dataframes
        """
        # 使用线程锁确保线程安全
        with ASRUtils.google_transcribe_lock:
            try:
                from google.cloud import speech_v1p1beta1
                from google.oauth2 import service_account

                # 从数据库获取ASR配置
                google_config = None
                db_config = EVSDataUtils.get_asr_config("google")

                if db_config and 'config' in db_config:
                    google_config = db_config['config']
                    logger.info(f"使用数据库中的Google配置: {google_config}")
                else:
                    # 如果数据库中没有配置，使用默认配置
                    logger.info("数据库中没有Google配置，使用默认配置")
                    from asr_config import ASR_CONFIG
                    google_config = ASR_CONFIG.get("google", {})

                # 使用传入的凭证文件路径
                if not credentials_file or not os.path.exists(credentials_file):
                    logger.error(f"Google API凭证文件不存在: {credentials_file}")
                    return pd.DataFrame(), pd.DataFrame()

                # 初始化Google客户端
                client = speech_v1p1beta1.SpeechClient.from_service_account_file(credentials_file)

                # 读取音频文件
                with io.open(file_path, "rb") as audio_file_obj:
                    content = audio_file_obj.read()

                # 检查使用的模型并设置配置
                available_models = google_config.get("available_models", ["default", "phone_call", "video", "command_and_search"])
                if model_name not in available_models:
                    logger.warning(f"模型 {model_name} 不在可用模型列表中 {available_models}，使用默认模型")
                    model_name = google_config.get("default_model", "default")

                # 检查语言代码映射
                language_mapping = google_config.get("language_mapping", {"zh": "zh-CN", "en": "en-US"})
                if language in language_mapping:
                    language = language_mapping[language]

                # 设置音频和识别配置
                audio = speech_v1p1beta1.RecognitionAudio(content=content)

                config = speech_v1p1beta1.RecognitionConfig(
                    encoding=speech_v1p1beta1.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code=language,
                    enable_word_time_offsets=True,
                    enable_automatic_punctuation=True,
                    model=model_name,
                )

                # 设置扬声器标记
                diarization_config = google_config.get("enable_speaker_diarization", False)
                if diarization_config:
                    config.enable_speaker_diarization = True
                    config.diarization_speaker_count = google_config.get("diarization_speaker_count", 2)

                # 请求转录
                response = client.recognize(config=config, audio=audio)

                # 准备数据库数据
                words_data = []
                segments_data = []

                # 处理转录结果
                for segment_id, result in enumerate(response.results):
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript

                    # 当可用时，处理扬声器标记
                    if hasattr(result, "speaker_tag"):
                        speaker = f"SPEAKER_{result.speaker_tag}"
                    else:
                        speaker = None

                    # 处理单词时间偏移
                    segment_start = float('inf')
                    segment_end = 0

                    for word_seq_no, word_info in enumerate(alternative.words):
                        word_text = word_info.word
                        word_start = word_info.start_time.total_seconds()
                        word_end = word_info.end_time.total_seconds()

                        # 更新段起始时间和结束时间
                        segment_start = min(segment_start, word_start)
                        segment_end = max(segment_end, word_end)

                        # 当可用时，添加扬声器标记
                        if hasattr(word_info, "speaker_tag"):
                            word_speaker = f"SPEAKER_{word_info.speaker_tag}"
                        else:
                            word_speaker = speaker

                        words_data.append({
                            'file_name': base_filename,
                            'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                            'lang': language,
                            'word': word_text,
                            'start_time': word_start,
                            'end_time': word_end,
                            'confidence': alternative.confidence,
                            'speaker': word_speaker,
                            'segment_id': segment_id,
                            'word_seq_no': word_seq_no,
                            'duration': word_end - word_start,
                            'slice_duration': duration,
                            'asr_provider': "google",
                            'model': model_name,
                            'channel_num': channel_num,
                            'audio_file': audio_file.replace(base_filename, f"{segment_id}.mp3")
                        })

                    segments_data.append({
                        'file_name': base_filename,
                        'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                        'lang': language,
                        'segment_id': segment_id,
                        'start_time': segment_start,
                        'end_time': segment_end,
                        'text': transcript,
                        'speaker': speaker,
                        'duration': segment_end - segment_start,
                        'slice_duration': duration,
                        'asr_provider': "google",
                        'model': model_name,
                        'channel_num': channel_num,
                        'audio_file': audio_file.replace(base_filename, f"{segment_id}.mp3")
                    })

                # 转换为DataFrame
                words_df = pd.DataFrame(words_data)
                segments_df = pd.DataFrame(segments_data)

                # 保存转录结果到文件
                output_prefix = os.path.splitext(base_filename)[0]
                result_file = os.path.join(EVS_RESOURCES_PATH, "slice_audio_files", output_prefix, "google", f"{output_prefix}_{model_name}_{language}_google.json")

                os.makedirs(os.path.dirname(result_file), exist_ok=True)

                with open(result_file, "w", encoding="utf-8") as f:
                    result_dict = {
                        "text": "\n".join([segment["text"] for segment in segments_data]),
                        "segments": segments_data,
                        "words": words_data
                    }
                    json.dump(result_dict, f, ensure_ascii=False, indent=4)

                return words_df, segments_df

            except Exception as e:
                logger.error(f"Error transcribing with Google: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return pd.DataFrame(), pd.DataFrame()

    @staticmethod
    def transcribe_with_ibm(file_path, asr_provider, language, base_filename, duration, audio_file, channel_num, model, api_key, service_url):
        """
        使用IBM Watson转录音频
        """
        try:
            # 读取音频文件
            with open(file_path, "rb") as audio_file:

                # Initialize IBM Watson Speech-to-Text client
                from ibm_watson import SpeechToTextV1
                from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

                if not api_key or not service_url:
                    logger.error("IBM Watson credentials not found")
                    return None, None

                # Set up authenticator and service
                authenticator = IAMAuthenticator(api_key)
                speech_to_text = SpeechToTextV1(authenticator=authenticator)
                speech_to_text.set_service_url(service_url)

                # Send request with proper error handling
                try:
                    response = speech_to_text.recognize(
                        audio=audio_file,
                        content_type="audio/mp3" if file_path.endswith('.mp3') else "audio/wav",
                        model=model,
                        timestamps=True,
                        word_confidence=True,
                    ).get_result()
                except Exception as e:
                    logger.error(f"IBM Watson API error: {str(e)}")
                    return pd.DataFrame(), pd.DataFrame()

                # 处理结果
                # Extract base filename from file_path
                base_filename = os.path.basename(file_path)

                # Prepare data containers for words and segments
                words_data = []
                segments_data = []

                # Process each result
                for idx, result in enumerate(response["results"]):
                    alternative = result["alternatives"][0]
                    transcript = alternative["transcript"]
                    confidence = alternative.get("confidence", 0)
                    timestamps = alternative.get("timestamps", [])

                    # Create segment ID
                    segment_id = f"{base_filename}_segment_{idx}"

                    # Get segment start and end times
                    segment_start = timestamps[0][1] if timestamps else 0
                    segment_end = timestamps[-1][2] if timestamps else 0

                    # Process words
                    for i, timestamp in enumerate(timestamps):
                        word = timestamp[0]
                        start_time = timestamp[1]
                        end_time = timestamp[2]

                        words_data.append({
                            'file_name': base_filename,
                            'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                            'segment_id': segment_id,
                            'word': word,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time,
                            'confidence': confidence,
                            'lang': language,
                            'asr_provider': asr_provider,
                            'model': model if isinstance(model, str) else str(model).split('(')[0],
                            'slice_duration': duration,
                            'channel_num': channel_num,
                            'audio_file': audio_file.replace(base_filename, f"{segment_id}.mp3")
                        })

                    # Add segment data
                    segments_data.append({
                        'file_name': base_filename,
                        'masked_file_name': ASRUtils.get_masked_filename(base_filename),
                        'lang': language,
                        'segment_id': segment_id,
                        'start_time': segment_start,
                        'end_time': segment_end,
                        'text': transcript,
                        'speaker': None,
                        'duration': segment_end - segment_start,
                        'asr_provider': asr_provider,
                        'model': model if isinstance(model, str) else str(model).split('(')[0],
                        'slice_duration': duration,
                        'channel_num': channel_num,
                        'audio_file': audio_file.replace(base_filename, f"{segment_id}.mp3")
                    })

                # Convert to DataFrame
                import pandas as pd
                words_df = pd.DataFrame(words_data)
                segments_df = pd.DataFrame(segments_data)

                return words_df, segments_df
        except Exception as e:
            logger.error(f"Error transcribing with IBM: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    @staticmethod
    def separate_audio_channels(file_path):
        """Separate stereo audio into two mono channels"""
        tmp_base = Path('c:/tmp')
        tmp_base.mkdir(parents=True, exist_ok=True)
        temp_dir = tmp_base / f'evs_audio_{time.strftime("%Y%m%d_%H%M%S")}'
        temp_dir.mkdir(exist_ok=True)

        audio = AudioSegment.from_file(file_path)
        channel1_path = temp_dir / 'channel1.wav'
        channel2_path = temp_dir / 'channel2.wav'

        if audio.channels == 1:
            is_mono = True
            audio.export(channel1_path, format='wav')
            channel2_path = None
        else:
            is_mono = False
            channel1 = audio.split_to_mono()[0]
            channel2 = audio.split_to_mono()[1]
            channel1.export(channel1_path, format='wav')
            channel2.export(channel2_path, format='wav')

        return channel1_path, channel2_path, temp_dir, is_mono

    @staticmethod
    def save_transcription_result(result, file_path, target_path, language):
        """
        保存转录结果
        """
        try:
            # 创建DataFrame
            df = pd.DataFrame(result)
            df['file_name'] = os.path.basename(file_path)
            df['language'] = language

            # 保存结果
            df.to_csv(target_path, index=False)

            return df
        except Exception as e:
            logger.error(f"Error saving transcription result: {str(e)}")
            return None

    @staticmethod
    def check_audio_channels(file_path):
        """
        Check number of channels in audio file
        """
        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                   '-show_entries', 'stream=channels', '-of', 'csv=p=0', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return int(result.stdout.strip())
        except Exception as e:
            print(f"Error checking audio channels: {str(e)}")
            return 0

    @staticmethod
    def process_uploaded_file(uploaded_file):
        """
        Process uploaded audio file, save to temporary directory
        """
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, "audio.wav")

            # Save uploaded file
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())

            return temp_dir
        except Exception as e:
            print(f"Error processing uploaded file: {str(e)}")
            return None

    @staticmethod
    def cleanup_temp_directory(temp_dir):
        """
        Clean up temporary directory
        """
        try:
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temp directory: {str(e)}")

    @staticmethod
    def process_audio_file_with_tencent(audio_path, target_path, secret_id, secret_key, language, channel_num=1):
        """
        Process audio file with Tencent Cloud ASR
        """
        print(f"Processing audio file with Tencent Cloud ASR, language: {language}")
        try:
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.asr.v20190614 import asr_client, models
            from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException

            # Set up Tencent Cloud API credentials
            cred = credential.Credential(secret_id, secret_key)

            # Set up HTTP configuration
            httpProfile = HttpProfile()
            httpProfile.endpoint = "asr.tencentcloudapi.com"

            # Set up client configuration
            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile

            # Create ASR client
            client = asr_client.AsrClient(cred, "ap-guangzhou", clientProfile)

            # Select engine model type
            engine_model_type = ""
            if language == "zh":
                engine_model_type = "16k_zh"
            elif language == "en":
                engine_model_type = "16k_en"
            else:
                engine_model_type = "16k_zh"  # Default to Chinese

            # Create request object
            req = models.CreateRecTaskRequest()

            # Set request parameters
            params = {
                "EngineModelType": engine_model_type,
                "ChannelNum": 1,
                "ResTextFormat": 2,
                "SourceType": 1,
                "Data": base64.b64encode(open(audio_path, "rb").read()).decode('utf-8')
            }

            req.from_json_string(json.dumps(params))

            # Send request
            resp = client.CreateRecTask(req)
            print(f"Full response: {resp}")

            # Try multiple methods to get TaskId
            task_id = None

            # Method 1: Direct access to TaskId property
            try:
                task_id = resp.TaskId
                print(f"Method 1 - TaskId: {task_id}")
            except Exception as e:
                print(f"Method 1 failed: {str(e)}")

            # Method 2: Access TaskId via Data.TaskId
            if task_id is None:
                try:
                    task_id = resp.Data.TaskId
                    print(f"Method 2 - TaskId: {task_id}")
                except Exception as e:
                    print(f"Method 2 failed: {str(e)}")

            # Method 3: Access TaskId via __dict__
            if task_id is None:
                try:
                    resp_dict = resp.Data.__dict__
                    task_id = resp_dict.get('TaskId')
                    print(f"Method 3 - TaskId: {task_id}")
                except Exception as e:
                    print(f"Method 3 failed: {str(e)}")

            # Method 4: Extract from JSON string
            if task_id is None:
                try:
                    resp_json = json.loads(resp.to_json_string())
                    task_id = resp_json.get('Data', {}).get('TaskId')
                    print(f"Method 4 - TaskId: {task_id}")
                except Exception as e:
                    print(f"Method 4 failed: {str(e)}")

            if task_id is None:
                print("Failed to retrieve TaskId from response")
                return None

            # Query task status
            status = 0
            result = None

            while status != 2:
                # Create query request
                describe_req = models.DescribeTaskStatusRequest()
                describe_req.TaskId = task_id

                # Send query request
                status_resp = client.DescribeTaskStatus(describe_req)
                print(f"Status response: {status_resp}")

                # Try multiple methods to get Status
                try:
                    status = status_resp.Status
                except Exception as e:
                    print(f"Direct Status access failed: {str(e)}")
                    try:
                        status = status_resp.Data.Status
                    except Exception as e:
                        print(f"Data.Status access failed: {str(e)}")
                        try:
                            status_json = json.loads(status_resp.to_json_string())
                            status = status_json.get('Data', {}).get('Status')
                        except Exception as e:
                            print(f"JSON Status access failed: {str(e)}")
                            status = 0

                print(f"Task status: {status}")

                # If task is completed
                if status == 2:
                    # Try multiple methods to get Result
                    try:
                        result = status_resp.Result
                    except Exception as e:
                        print(f"Direct Result access failed: {str(e)}")
                        try:
                            result = status_resp.Data.Result
                        except Exception as e:
                            print(f"Data.Result access failed: {str(e)}")
                            try:
                                result_json = json.loads(status_resp.to_json_string())
                                result = result_json.get('Data', {}).get('Result')
                            except Exception as e:
                                print(f"JSON Result access failed: {str(e)}")
                                result = None

                # If task failed
                elif status == 3:
                    print("Task failed")
                    return None

                # Wait before querying again
                import time
                time.sleep(2)

            # Save result
            if result:
                with open(target_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                return result
            else:
                print("No result returned")
                return None

        except TencentCloudSDKException as err:
            print(f"Tencent Cloud SDK error: {err}")
            return None
        except Exception as e:
            print(f"Error processing audio with Tencent: {str(e)}")
            return None

    @staticmethod
    def slice_audio_file(input_file, duration, asr_provider, model_folder):
        """
        Slice audio file into segments of specified duration

        Args:
            input_file: Input audio file
            duration: Duration of each segment (seconds)

        Returns:
            Output directory path
        """
        try:
            # # Ensure base directory exists
            # if not os.path.exists(EVS_RESOURCES_PATH):
            #     os.makedirs(EVS_RESOURCES_PATH)
            #     logger.info(f"Created base directory {EVS_RESOURCES_PATH}")

            # # Ensure slice_audio_files directory exists
            # slice_audio_dir = os.path.join(EVS_RESOURCES_PATH, "slice_audio_files")
            # if not os.path.exists(slice_audio_dir):
            #     os.makedirs(slice_audio_dir)
            #     logger.info(f"Created slice audio directory {slice_audio_dir}")

            audio = AudioSegment.from_file(input_file)
            slice_duration = duration * 1000  # Convert to milliseconds

            output_prefix = os.path.splitext(input_file.name)[0]
            output_path = os.path.join(EVS_RESOURCES_PATH, "slice_audio_files", output_prefix, asr_provider, model_folder if model_folder else "default")

            # Create output directory
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                logger.info(f"Created output directory {output_path}")
            else: # remove existed files in the output directory
                logger.info(f"Removed existing files in {output_path}")
                for file in os.listdir(output_path):
                    os.unlink(os.path.join(output_path, file))

            # Use Streamlit progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_chunks = len(audio) // slice_duration + (1 if len(audio) % slice_duration > 0 else 0)

            for i in range(0, len(audio), slice_duration):
                chunk = audio[i:i + slice_duration]
                chunk_number = i // slice_duration

                # Update progress
                progress = int((chunk_number + 1) / total_chunks * 100)
                progress_bar.progress(progress)
                status_text.write(f"Processing audio segment {chunk_number + 1}/{total_chunks}")

                # Export audio segment
                chunk_path = os.path.join(output_path, f"{chunk_number}.mp3")
                chunk.export(chunk_path, format="mp3")
                logger.info(f"Exported chunk {chunk_number} to {chunk_path}")

            # Complete processing
            progress_bar.progress(100)
            status_text.write("Audio slicing completed!")
            time.sleep(0.5)  # Brief display of completion message
            status_text.empty()

            return output_path

        except Exception as e:
            logger.error(f"Error slicing audio: {str(e)}")
            raise e

    @staticmethod
    def convert_flac_to_mp3(input_file, asr_provider, model_folder):
        """
        Convert FLAC audio file to MP3 format

        Args:
            input_file: Input FLAC audio file (can be Path object, filename, or uploaded file object)

        Returns:
            Path to the converted MP3 file
        """
        try:
            # 添加额外的输入检查
            if input_file is None:
                error_msg = "No file provided for conversion"
                logger.error(error_msg)
                st.error(error_msg)
                return None

            # Ensure base directory exists
            if not os.path.exists(EVS_RESOURCES_PATH):
                os.makedirs(EVS_RESOURCES_PATH)
                logger.info(f"Created base directory {EVS_RESOURCES_PATH}")

            # Ensure converted files directory exists
            converted_dir = os.path.join(EVS_RESOURCES_PATH, "converted_files")
            if not os.path.exists(converted_dir):
                os.makedirs(converted_dir)
                logger.info(f"Created converted files directory {converted_dir}")

            # Handle different types of input files
            temp_file = None

            # Check if it's a Streamlit UploadedFile object
            if hasattr(input_file, 'name') and hasattr(input_file, 'getbuffer'):
                input_filename = input_file.name
                if not input_filename:
                    error_msg = "Upload file has no name"
                    logger.error(error_msg)
                    st.error(error_msg)
                    return None
                # Save uploaded file to temporary directory
                logger.info(f"Processing uploaded file: {input_filename}")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(input_filename)[1])
                temp_file.write(input_file.getbuffer())
                temp_file.close()
                input_filepath = temp_file.name
            # Handle Path object or string path
            elif hasattr(input_file, 'name') or isinstance(input_file, str):
                if hasattr(input_file, 'name'):
                    input_filename = input_file.name
                    input_filepath = str(input_file)
                else:
                    input_filename = os.path.basename(input_file)
                    input_filepath = input_file
            else:
                error_msg = f"Unsupported file type: {type(input_file)}"
                logger.error(error_msg)
                st.error(error_msg)
                return None

            # Get filename without extension
            filename_without_ext = os.path.splitext(input_filename)[0]

            # Build output file path
            output_filepath = os.path.join(converted_dir, f"{filename_without_ext}.mp3")

            # Load audio file using pydub
            st.info(f"Converting file: {input_filename}")
            with st.spinner("Converting..."):
                logger.info(f"Loading audio from {input_filepath}")
                audio = AudioSegment.from_file(input_filepath)

                # Export as MP3 format
                logger.info(f"Exporting audio to MP3 format: {output_filepath}")
                audio.export(output_filepath, format="mp3")

            st.success(f"File successfully converted to MP3 format")
            logger.info(f"File converted: {input_filepath} -> {output_filepath}")

            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
                logger.info(f"Deleted temporary file: {temp_file.name}")

            return output_filepath

        except Exception as e:
            error_msg = f"Failed to convert audio file: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)

            # Ensure temporary file is cleaned up
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                    logger.info(f"Deleted temporary file: {temp_file.name}")
                except OSError as cleanup_error:
                    logger.debug(f"Failed to cleanup temp file: {cleanup_error}")

            return None