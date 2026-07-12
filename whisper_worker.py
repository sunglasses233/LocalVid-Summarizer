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

# ================= 统一配置 =================
from config import (
    AUDIO_WORKSPACE_DIR,
    EN_ABSOLUTE_MAX_CHARS,
    EN_MAX_CHARS,
    EN_MAX_GAP,
    EN_TOLERANCE_WORDS,
    HF_ENDPOINT,
    VOCAL_MODEL_DIR,
    VOCAL_SEPARATOR_MODEL_NAME,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_DEVICE,
    WHISPER_LOCAL_FILES_ONLY,
    WHISPER_MIN_SILENCE_MS,
    WHISPER_MODEL_DIR,
    WHISPER_MODEL_NAME,
    WHISPER_VAD_FILTER,
    WHISPER_WORD_TIMESTAMPS,
)

os.environ["HF_ENDPOINT"] = HF_ENDPOINT

def format_time_srt(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{math.floor(seconds):02d},{milliseconds:03d}"

# [编程大师2.0 新增]：按需路由降噪策略
# [编程大师2.0 终极修复]：按需路由降噪策略
def extract_and_sanitize_audio(file_path, use_denoise=False):
    import tempfile # [核心修复]：引入 tempfile 获取标准临时目录
    import shutil
    
    if not shutil.which("ffmpeg"): 
        print("❌ 致命错误: 未找到 ffmpeg，无法处理音频！")
        return file_path, False

    custom_workspace = str(AUDIO_WORKSPACE_DIR)
    os.makedirs(custom_workspace, exist_ok=True)
    tempfile.tempdir = custom_workspace # 强行修改 Python 的默认临时目录
    
    workspace = Path(custom_workspace)
    
    temp_input_wav = workspace / f"{file_path.stem}_temp_raw.wav"
    final_wav_path = workspace / f"{file_path.stem}_temp_sanitized.wav"
    
    # ================= 策略 1：极速基础模式（用户未开启深度降噪） =================
    if not use_denoise:
        print(f"\n🎛️ [预处理] 基础提取模式 (用户未开启深度降噪): {file_path.name}")
        command = ["ffmpeg", "-y", "-i", str(file_path), "-vn", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", "-af", "aresample=async=1", str(final_wav_path)]
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return final_wav_path, True
        except subprocess.CalledProcessError:
            print("[Whisper Worker] ⚠️ FFmpeg 提取失败，将尝试直接读取原文件。", flush=True)
            if final_wav_path.exists():
                try: final_wav_path.unlink()
                except: pass
            return file_path, False

    # ================= 策略 2：深度降噪模式（AI 人声分离） =================
    print(f"\n🎛️ [预处理] 启动深度降噪流，正在进行本地解耦与 AI 人声分离: {file_path.name}")
    
    # === 步骤 1：FFmpeg 抽轨 ===
    print(f"⏳ [步骤 1/3] 正在从视频容器中抽取音轨至本地 ({workspace})...")
    try:
        command_extract = [
            "ffmpeg", "-y", "-i", str(file_path), 
            "-vn", "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2", 
            str(temp_input_wav)
        ]
        subprocess.run(command_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if not temp_input_wav.exists() or temp_input_wav.stat().st_size < 10240:
            print("⚠️ 警告: 提取出的初始音频过小，FFmpeg 可能抓取到了空音轨！")
            return file_path, False

    except subprocess.CalledProcessError as e:
        print(f"❌ 步骤 1 失败: FFmpeg 无法读取音频流！详细错误: {e}")
        return file_path, False

    # === 步骤 2：进行 AI 噪音剥离 ===
    print("⏳ [步骤 2/3] 正在通过本地算力进行 AI 噪音剥离...")
    try:
        from audio_separator.separator import Separator

        VOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        separator = Separator(
            output_dir=str(workspace),
            output_format="WAV",
            model_file_dir=str(VOCAL_MODEL_DIR)
        )

        separator.load_model(model_filename=VOCAL_SEPARATOR_MODEL_NAME)

        out_file1, out_file2 = separator.separate(str(temp_input_wav))
        
        if "(Vocals)" in out_file1:
            vocals_file = out_file1
            inst_file = out_file2
        else:
            vocals_file = out_file2
            inst_file = out_file1

        # 拼装工作区绝对路径
        vocals_full_path = workspace / vocals_file
        inst_full_path = workspace / inst_file

        # === 步骤 3：本地重采样 ===
        print(f"🔄 [步骤 3/3] 人声分离完成，正在本地重采样为 Whisper 基准频率...")
        command_resample = [
            "ffmpeg", "-y", "-i", str(vocals_full_path),
            "-vn", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(final_wav_path)
        ]
        subprocess.run(command_resample, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # 【文件清理】处理完毕，立即释放磁盘空间
        if temp_input_wav.exists(): temp_input_wav.unlink() 
        if inst_full_path.exists(): inst_full_path.unlink() 
        if vocals_full_path.exists(): vocals_full_path.unlink() 

        del separator
        import gc
        gc.collect()

        return final_wav_path, True

    except Exception as e:
        print(f"⚠️ 人声分离过程遭遇异常: {e}")
        import traceback
        traceback.print_exc()
        if temp_input_wav.exists(): temp_input_wav.unlink()
        print("⏬ 降级退回基础 FFmpeg 音轨提取模式...")
    
        # 【容错降级机制】如果人声分离崩溃，自动退回策略1
        command = ["ffmpeg", "-y", "-i", str(file_path), "-vn", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", "-af", "aresample=async=1", str(final_wav_path)]
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return final_wav_path, True
        except subprocess.CalledProcessError:
            return file_path, False

    except Exception as e:
        print(f"⚠️ 人声分离过程遭遇异常: {e}")
        import traceback
        traceback.print_exc()
        if temp_input_wav.exists(): temp_input_wav.unlink()
        print("⏬ 降级退回基础 FFmpeg 音轨提取模式...")
    
        # 【容错降级机制】如果人声分离崩溃，自动退回策略1
        command = ["ffmpeg", "-y", "-i", str(file_path), "-vn", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", "-af", "aresample=async=1", str(final_wav_path)]
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return final_wav_path, True
        except subprocess.CalledProcessError:
            return file_path, False

def emit_progress(current_time, total_duration):
    """【新增】向主进程广播当前进度的核心函数"""
    if total_duration > 0:
        percent = min(100.0, (current_time / total_duration) * 100)
        # 强制格式化输出并立刻冲刷缓冲区，确保 app.py 瞬间捕获
        print(f"[PROGRESS] {percent:.1f}", flush=True)

def transcribe_and_save(audio_path, output_json_path, model_size=WHISPER_MODEL_NAME, device=WHISPER_DEVICE, use_denoise=False):
    print(f"[Whisper Worker] 正在加载模型 {model_size} 到 {device}...", flush=True)
    try:
        model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=WHISPER_COMPUTE_TYPE, 
            download_root=str(WHISPER_MODEL_DIR),
            local_files_only=WHISPER_LOCAL_FILES_ONLY
        )
    except Exception as e:
        print(f"[Whisper Worker] 模型加载失败: {e}", flush=True)
        sys.exit(1)

    file_path = Path(audio_path)
    audio_target, is_temp = extract_and_sanitize_audio(file_path, use_denoise)

    print(f"[Whisper Worker] 开始转录...", flush=True)
    try:
        segments_generator, info = model.transcribe(
            str(audio_target),
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=WHISPER_VAD_FILTER,
            vad_parameters=dict(min_silence_duration_ms=WHISPER_MIN_SILENCE_MS),
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            word_timestamps=WHISPER_WORD_TIMESTAMPS 
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
    parser.add_argument("--model", type=str, default=WHISPER_MODEL_NAME)
    # 新增接收参数
    parser.add_argument("--denoise", action="store_true", help="启用 AI 人声分离降噪")

        
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[Whisper Worker] 错误：找不到输入文件 {args.input}", flush=True)
        sys.exit(1)
        
    # 透传给主函数
    transcribe_and_save(args.input, args.output, args.model, use_denoise=args.denoise)
    os._exit(0)