import logging
import os
import shutil
import subprocess
import torch

def ffmpeg_extract_audio_by_fps(video_path, output_path, fps, audio_format="wav", segment_duration=None):
    """
    按指定FPS提取视频中的音频
    
    参数:
        video_path (str): 视频文件路径
        output_path (str): 输出音频文件路径
        fps (int/float): 目标帧率(每秒帧数)
        audio_format (str): 音频格式 (wav, mp3, aac, flac)
        segment_duration (float): 分段提取时长(秒)，None表示提取完整音频
    """
    # 验证FPS值
    if fps <= 0:
        raise ValueError("FPS值必须大于0")
    
    # 根据格式选择编码器
    codec_map = {
        "mp3": "libmp3lame",
        "wav": "pcm_s16le",
        "aac": "aac",
        "flac": "flac"
    }
    
    if audio_format not in codec_map:
        raise ValueError(f"不支持的格式: {audio_format}")
    
    try:
        # 计算采样率 (通常使用48kHz或44.1kHz，但按FPS调整)
        sample_rate = 48000  # 标准采样率
        frame_duration = 1.0 / fps  # 每帧持续时间(秒)
        
        # 构建基础FFmpeg命令
        cmd = [
            "ffmpeg",
            "-i", video_path,        # 输入文件
            "-vn",                   # 禁用视频流
            "-acodec", codec_map[audio_format],  # 音频编码器
            "-q:a", "0",             # 最高质量
            "-y"                     # 覆盖输出文件
        ]
        
        # 如果指定了分段时长
        if segment_duration:
            # 计算分段数量
            duration = get_video_duration(video_path)
            segments = int(duration / segment_duration) + 1
            
            # 创建输出目录
            os.makedirs(output_path, exist_ok=True)
            
            # 分段提取音频
            for i in range(segments):
                start_time = i * segment_duration
                segment_output = os.path.join(output_path, f"segment_{i+1}.{audio_format}")
                
                segment_cmd = cmd + [
                    "-ss", str(start_time),      # 开始时间
                    "-t", str(segment_duration), # 持续时间
                    segment_output
                ]
                
                subprocess.run(segment_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"✅ 音频分段 {i+1}/{segments} 保存成功: {segment_output}")
            
            return output_path
        
        # 完整音频提取
        cmd.append(output_path)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ FFmpeg音频保存成功: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        print(f"❌ FFmpeg执行失败: {error_msg}")
        return None

def get_video_duration(video_path):
    """获取视频时长(秒)"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return float(result.stdout.strip())
    else:
        raise RuntimeError(f"无法获取视频时长: {result.stderr}")

def ffmpeg_extract_audio(video_path, output_path, audio_format="wav"):
    """使用FFmpeg提取音频"""
    
    # 根据格式选择编码器
    codec_map = {
        "mp3": "libmp3lame",
        "wav": "pcm_s16le",
        "aac": "aac",
        "flac": "flac"
    }
    
    if audio_format not in codec_map:
        raise ValueError(f"不支持的格式: {audio_format}")
    
    try:
        # 构建FFmpeg命令
        cmd = [
            "ffmpeg",
            "-i", video_path,        # 输入文件
            "-vn",                   # 禁用视频流
            "-acodec", codec_map[audio_format],  # 音频编码器
            "-q:a", "0",             # 最高质量 (MP3)
            "-y",                    # 覆盖输出文件
            output_path
        ]
        
        # 执行命令（隐藏输出）
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ FFmpeg音频保存成功: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg执行失败: {e.stderr.decode() if e.stderr else str(e)}")
        return None


def merge_video_audio(video_path: str, audio_path: str):
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """
    # set logging
    logging.basicConfig(level=logging.INFO)

    # check
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        # create ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # overwrite
            '-i',
            video_path,
            '-i',
            audio_path,
            '-c:v',
            'copy',  # copy video stream
            '-c:a',
            'aac',  # use AAC audio encoder
            '-b:a',
            '192k',  # set audio bitrate (optional)
            '-map',
            '0:v:0',  # select the first video stream
            '-map',
            '1:a:0',  # select the first audio stream
            '-shortest',  # choose the shortest duration
            temp_output
        ]

        # execute the command
        logging.info("Start merging video and audio...")
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # check result
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        logging.info(f"Merge completed, saved to {video_path}")

    except Exception as e:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        logging.error(f"merge_video_audio failed with error: {e}")