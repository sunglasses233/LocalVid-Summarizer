import os
import sys
import json
import math
import argparse
import traceback
import subprocess
import shutil
from pathlib import Path
from faster_whisper import WhisperModel

# 强制使用 Hugging Face 国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ================= 英语/表音文字 断句参数配置 =================
EN_MAX_CHARS = 70           
EN_ABSOLUTE_MAX_CHARS = 90  
EN_MAX_GAP = 0.8            
EN_TOLERANCE_WORDS = 2      

def format_time_srt(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{math.floor(seconds):02d},{milliseconds:03d}"

def extract_and_sanitize_audio(file_path):
    if not shutil.which("ffmpeg"): 
        print("[Whisper Worker] ⚠️ 未检测到 FFmpeg，跳过音轨净化。", flush=True)
        return file_path, False
        
    temp_wav_path = file_path.with_name(f"{file_path.stem}_temp_sanitized.wav")
    print(f"[Whisper Worker] 🎛️ 正在净化重采样音轨: {file_path.name}", flush=True)
    command = [
        "ffmpeg", "-y", "-i", str(file_path), 
        "-vn", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", 
        "-af", "aresample=async=1", str(temp_wav_path)
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return temp_wav_path, True
    except subprocess.CalledProcessError:
        print("[Whisper Worker] ⚠️ FFmpeg 提取失败，将尝试直接读取原文件。", flush=True)
        if temp_wav_path.exists():
            try: temp_wav_path.unlink()
            except: pass
        return file_path, False

def emit_progress(current_time, total_duration):
    """【新增】向主进程广播当前进度的核心函数"""
    if total_duration > 0:
        percent = min(100.0, (current_time / total_duration) * 100)
        # 强制格式化输出并立刻冲刷缓冲区，确保 app.py 瞬间捕获
        print(f"[PROGRESS] {percent:.1f}", flush=True)

def transcribe_and_save(audio_path, output_json_path, model_size="large-v3", device="cuda"):
    print(f"[Whisper Worker] 正在加载模型 {model_size} 到 {device}...", flush=True)
    try:
        model = WhisperModel(
            model_size, 
            device=device, 
            compute_type="float16", 
            download_root=r"G:\WhisperModels", 
            local_files_only=True
        )
    except Exception as e:
        print(f"[Whisper Worker] 模型加载失败: {e}", flush=True)
        sys.exit(1)

    file_path = Path(audio_path)
    audio_target, is_temp = extract_and_sanitize_audio(file_path)

    print(f"[Whisper Worker] 开始转录...", flush=True)
    try:
        segments_generator, info = model.transcribe(
            str(audio_target),
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
            word_timestamps=True 
        )
        
        total_duration = info.duration # 获取音频总秒数
        lang_code = info.language.lower()
        print(f"[Whisper Worker] 🌐 检测到语言: {lang_code} (置信度: {info.language_probability:.2f}, 时长: {total_duration:.1f}s)", flush=True)
        
        subtitles = []
        full_text_list = []
        seg_id = 1
        
        # ================= 🚀 双引擎路由逻辑 =================
        if lang_code.startswith('zh') or lang_code == 'ja':
            print("[Whisper Worker] ⏳ 路由策略: CJK 原生段落引擎...", flush=True) 
            for segment in segments_generator:
                text = segment.text.strip()
                full_text_list.append(text)
                subtitles.append({
                    'id': str(seg_id),
                    'start_time': format_time_srt(segment.start),
                    'end_time': format_time_srt(segment.end),
                    'text': text
                })  
                seg_id += 1
                emit_progress(segment.end, total_duration) # 广播进度
        else:
            print("[Whisper Worker] ⏳ 路由策略: 英语/表音文字 段落级前瞻容忍引擎...", flush=True) 
            for segment in segments_generator:
                if segment.words:
                    current_words = []
                    chunk_start = segment.words[0].start
                    
                    for i, word_obj in enumerate(segment.words):
                        current_words.append(word_obj.word)
                        is_last_word = (i == len(segment.words) - 1)
                        words_left = len(segment.words) - 1 - i  
                        
                        gap_to_next = 0.0
                        if not is_last_word:
                            gap_to_next = segment.words[i+1].start - word_obj.end
                            
                        current_text = "".join(current_words).strip()
                        must_cut = is_last_word or len(current_text) >= EN_ABSOLUTE_MAX_CHARS
                        
                        if not must_cut and (len(current_text) >= EN_MAX_CHARS or gap_to_next >= EN_MAX_GAP):
                            if words_left > EN_TOLERANCE_WORDS:
                                must_cut = True
                        
                        if must_cut:
                            chunk_end = word_obj.end
                            full_text_list.append(current_text)
                            subtitles.append({
                                'id': str(seg_id),
                                'start_time': format_time_srt(chunk_start),
                                'end_time': format_time_srt(chunk_end),
                                'text': current_text
                            })
                            seg_id += 1
                            if not is_last_word:
                                chunk_start = segment.words[i+1].start
                            current_words = []
                            emit_progress(chunk_end, total_duration) # 广播局部进度
                else:
                    text = segment.text.strip()
                    full_text_list.append(text)
                    subtitles.append({
                        'id': str(seg_id),
                        'start_time': format_time_srt(segment.start),
                        'end_time': format_time_srt(segment.end),
                        'text': text
                    })  
                    seg_id += 1
                    emit_progress(segment.end, total_duration) # 广播进度

        # 确保最后进度达到 100%
        print("[PROGRESS] 100.0", flush=True)

        if is_temp and audio_target.exists():
            try: audio_target.unlink()
            except: pass

        output_data = {
            "language": lang_code,
            "full_text": "\n".join(full_text_list), 
            "subtitles": subtitles
        }
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        print(f"[Whisper Worker] ✅ 转录成功，结果已保存至: {output_json_path}", flush=True)
        
    except Exception as e:
        print(f"[Whisper Worker] ❌ 转录过程中发生致命错误:", flush=True)
        traceback.print_exc()
        if is_temp and audio_target.exists():
            try: audio_target.unlink()
            except: pass
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立运行的 Whisper 转录进程")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="large-v3")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[Whisper Worker] 错误：找不到输入文件 {args.input}", flush=True)
        sys.exit(1)
        
    transcribe_and_save(args.input, args.output, args.model)
    os._exit(0)