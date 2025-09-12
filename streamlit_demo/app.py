# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import base64
import datetime
import hashlib
import json
import os
import random
import re
import sys
import subprocess
import urllib.parse
# from streamlit_js_eval import streamlit_js_eval
from functools import partial
from io import BytesIO
import tempfile

import cv2
import numpy as np
import requests
import streamlit as st
from constants import LOGDIR, server_error_msg
from library import Library
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_select import image_select
import imageio_ffmpeg

custom_args = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--controller_url', type=str, default='http://localhost:40000', help='url of the controller')
parser.add_argument('--sd_worker_url', type=str, default='http://0.0.0.0:40006', help='url of the stable diffusion worker')
parser.add_argument('--chatgpt_worker_url', type=str, default='http://localhost:40002', help='url of the chatgpt worker')
parser.add_argument('--max_image_limit', type=int, default=4, help='maximum number of images')
parser.add_argument('--use_chatgpt', action='store_true', help='use ChatGPT instead of local model')
args = parser.parse_args(custom_args)
controller_url = args.controller_url
sd_worker_url = args.sd_worker_url
chatgpt_worker_url = args.chatgpt_worker_url
max_image_limit = args.max_image_limit
use_chatgpt = args.use_chatgpt
print('args:', args)


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f'{t.year}-{t.month:02d}-{t.day:02d}-conv.json')
    return name


def get_model_list():
    if use_chatgpt:
        return ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo']
    
    ret = requests.post(controller_url + '/refresh_all_workers')
    assert ret.status_code == 200
    ret = requests.post(controller_url + '/list_models')
    models = ret.json()['models']
    models = [item for item in models if 'InternVL2-Det' not in item and 'InternVL2-Gen' not in item]
    return models


def is_video_url(url):
    """检查是否为视频链接"""
    video_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/',
        r'(?:https?://)?youtu\.be/',
        r'(?:https?://)?(?:www\.)?vimeo\.com/',
        r'(?:https?://)?(?:www\.)?bilibili\.com/',
        r'(?:https?://)?(?:www\.)?dailymotion\.com/',
        r'(?:https?://)?(?:www\.)?twitch\.tv/',
        r'\.mp4(?:\?.*)?$',
        r'\.mov(?:\?.*)?$',
        r'\.avi(?:\?.*)?$',
        r'\.mkv(?:\?.*)?$',
        r'\.webm(?:\?.*)?$',
    ]
    return any(re.search(pattern, url) for pattern in video_patterns)


def get_video_info(url):
    """获取视频信息而不下载"""
    try:
        cmd = [
            # 'C:\\Users\\PDLP-013-Eric\\Anaconda3\\envs\\video\\python.exe', '-m', 'yt_dlp',
            'python', '-m', 'yt_dlp',
            '--dump-json',
            '--no-playlist',
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise Exception(f"获取视频信息失败: {result.stderr}")
        
        info = json.loads(result.stdout)
        return info
        
    except subprocess.TimeoutExpired:
        raise Exception("获取视频信息超时")
    except Exception as e:
        raise Exception(f"获取视频信息时出错: {str(e)}")


def extract_video_subtitles(url):
    """提取视频字幕"""
    try:
        # 首先检查可用的字幕
        cmd = [
            'yt-dlp',
            '--list-subs',
            '--no-playlist',
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return None
        
        # 检查是否有字幕可用
        if "Available automatic captions" in result.stdout or "Available subtitles" in result.stdout:
            # 有字幕可用，继续处理
            pass
        elif "has no subtitles" in result.stdout or "No subtitles" in result.stdout:
            return None
        else:
            # 如果没有明确的字幕信息，也尝试提取
            pass
        
        # 提取字幕（优先选择英文和中文）
        subtitle_cmd = [
            'yt-dlp',
            '--write-subs',
            '--write-auto-subs',
            '--sub-langs', 'en,zh,zh-cn,zh-tw',
            '--sub-format', 'vtt',
            '--skip-download',
            '--no-playlist',
            '--output', '/tmp/%(title)s.%(ext)s',
            url
        ]
        
        result = subprocess.run(subtitle_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return None
        
        # 查找生成的字幕文件
        import glob
        subtitle_files = glob.glob('/tmp/*.vtt')
        
        if not subtitle_files:
            return None
        
        # 读取字幕内容
        subtitles = []
        for subtitle_file in subtitle_files:
            try:
                with open(subtitle_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    subtitles.append({
                        'file': subtitle_file,
                        'content': content
                    })
            except Exception as e:
                continue
        
        return subtitles if subtitles else None
        
    except Exception as e:
        return None


def stream_video_frames(url):
    """流式处理视频帧，不下载整个文件"""
    try:
        # 获取视频信息
        with st.spinner('Getting video information...'):
            video_info = get_video_info(url)
        
        duration = video_info.get('duration', 0)
        title = video_info.get('title', 'Unknown')
        
        # 尝试提取字幕
        subtitles = None
        with st.spinner('Extracting subtitles...'):
            subtitles = extract_video_subtitles(url)
            if subtitles:
                st.info(f"📝 Found {len(subtitles)} subtitle file(s)")
            else:
                st.info("📝 No subtitles available for this video")
        
        # 使用yt-dlp获取最佳视频流URL，然后用ffmpeg处理
        with st.spinner('Getting video stream URL...'):
            cmd = [
                # 'C:\\Users\\PDLP-013-Eric\\Anaconda3\\envs\\video\\python.exe', '-m', 'yt_dlp',
                'python', '-m', 'yt_dlp',
                '-f', 'best[height<=720]',  # 选择720p以下的视频流
                '--get-url',
                '--no-playlist',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise Exception(f"获取视频流URL失败: {result.stderr}")
            
            stream_url = result.stdout.strip()
            if not stream_url:
                raise Exception("无法获取视频流URL")
        
        # 使用ffmpeg从流URL中提取帧
        with st.spinner(f'Processing video: {title}...'):
            # 根据视频长度决定采样率
            if duration <= 30:
                fps_filter = 'fps=2'  # 短视频：每秒2帧
                max_frames = min(60, int(duration * 2))  # 最多60帧
            elif duration <= 120:
                fps_filter = 'fps=1'  # 中等视频：每秒1帧
                max_frames = min(120, int(duration))  # 最多120帧
            else:
                fps_filter = 'fps=1'  # 长视频：每秒1帧，不设置上限
                max_frames = int(duration)  # 根据时长动态设置
            
            # 获取ffmpeg的完整路径
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_path,
                '-loglevel', 'error',  # 只显示错误信息，隐藏警告
                '-i', stream_url,
                '-vf', fps_filter,
                '-f', 'image2pipe',
                '-vcodec', 'png',
                '-frames:v', str(max_frames),  # 动态设置帧数
                '-'
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            frames = []
            frame_count = 0
            
            # 使用更简单的方法读取PNG数据
            try:
                # 读取所有输出数据
                stdout_data, stderr_data = process.communicate(timeout=60)
                
                # 隐藏ffmpeg的stderr输出，不显示警告信息
                # if stderr_data:
                #     st.warning(f"ffmpeg警告: {stderr_data.decode()[:200]}")
                
                if stdout_data:
                    # 查找PNG文件头
                    png_start = b'\x89PNG\r\n\x1a\n'
                    png_end = b'IEND\xaeB`\x82'
                    
                    data = stdout_data
                    start = 0
                    
                    while True:
                        # 查找PNG开始标记
                        png_start_pos = data.find(png_start, start)
                        if png_start_pos == -1:
                            break
                        
                        # 查找PNG结束标记
                        png_end_pos = data.find(png_end, png_start_pos)
                        if png_end_pos == -1:
                            break
                        
                        # 提取PNG数据
                        png_data = data[png_start_pos:png_end_pos + len(png_end)]
                        
                        try:
                            img = Image.open(BytesIO(png_data))
                            frames.append(img)
                            frame_count += 1
                            
                            if frame_count >= max_frames:  # 使用动态帧数限制
                                break
                                
                        except Exception as e:
                            pass
                        
                        start = png_end_pos + len(png_end)
                
            except subprocess.TimeoutExpired:
                process.kill()
                st.warning("Video processing timeout")
            except Exception as e:
                st.warning(f"Error processing video frames: {str(e)}")
        
        if frames:
            subtitle_info = ""
            if subtitles:
                subtitle_info = f" and {len(subtitles)} subtitle file(s)"
            st.success(f"🎬 Video processing completed: Extracted {len(frames)} frames{subtitle_info} from {title} (Duration: {duration:.1f}s)")
        else:
            st.warning("Failed to extract frames from video, please check if the link is valid or try another video")
        
        return {
            'frames': frames,
            'subtitles': subtitles,
            'title': title,
            'duration': duration
        }
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return []


def process_video_url(url):
    """处理视频链接 - 使用流式处理"""
    result = stream_video_frames(url)
    if isinstance(result, dict):
        return result['frames']
    return result


def load_upload_file_and_show():
    images, filenames = [], []
    # 对每个加入的图像记录是否需要持久化到磁盘（普通图片：True；视频帧：False）
    persist_flags = []
    
    # 视频帧单独处理，不添加到images列表中
    video_frames_for_ai = []
    if 'video_frames' in st.session_state and st.session_state.video_frames:
        video_frames_for_ai = st.session_state.video_frames.copy()
    
    # 处理上传的文件
    if uploaded_files is not None:
        video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
        def is_video_file(name, mime_type):
            ext = os.path.splitext(name)[1].lower()
            return (mime_type and mime_type.startswith('video')) or (ext in video_exts)

        def extract_video_frames_to_pil(tmp_video_path, max_frames=None):
            cap = cv2.VideoCapture(tmp_video_path)
            if not cap.isOpened():
                return []
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if total <= 0:
                total = 1
            if fps <= 0:
                fps = 1
            
            # 智能关键帧选择策略
            if max_frames is None:
                # 完全动态选择：根据视频长度和内容智能决定
                duration = total / fps
                
                if total <= 30:
                    # 短视频：抽取所有帧
                    target_frames = total
                elif duration <= 5:
                    # 5秒内：每0.1秒一帧
                    target_frames = min(total, int(duration * 10))
                elif duration <= 30:
                    # 30秒内：每0.2秒一帧
                    target_frames = min(total, int(duration * 5))
                elif duration <= 120:
                    # 2分钟内：每0.5秒一帧
                    target_frames = min(total, int(duration * 2))
                else:
                    # 长视频：每1秒一帧，不设置上限
                    target_frames = int(duration)
                
                # 使用智能采样选择关键帧
                if target_frames >= total:
                    # 抽取所有帧
                    frames = []
                    while True:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(rgb))
                else:
                    # 智能关键帧选择：选择变化最大的帧
                    frames = []
                    frame_indices = []
                    
                    # 先均匀采样候选帧
                    candidate_indices = np.linspace(0, total - 1, min(target_frames * 3, total), dtype=int)
                    
                    # 计算帧间差异，选择变化最大的帧
                    prev_frame = None
                    frame_diffs = []
                    
                    for idx in candidate_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            if prev_frame is not None:
                                diff = cv2.absdiff(gray, prev_frame)
                                frame_diffs.append((idx, np.mean(diff)))
                            prev_frame = gray
                    
                    # 选择差异最大的帧
                    frame_diffs.sort(key=lambda x: x[1], reverse=True)
                    selected_indices = [x[0] for x in frame_diffs[:target_frames]]
                    selected_indices.sort()
                    
                    # 提取选中的帧
                    for idx in selected_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(Image.fromarray(rgb))
            else:
                # 如果指定了max_frames，使用均匀采样
                n = max(1, min(max_frames, total))
                ids = np.linspace(0, total - 1, n, dtype=np.int32)
                frames = []
                for fid in ids:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(rgb))
            cap.release()
            return frames

        # 计算当前会话中已使用的图片配额（用户上传的图像/帧）
        try:
            used_images = 0
            for m in st.session_state.get('messages', []):
                if m.get('role') == 'user' and 'image' in m:
                    used_images += len(m['image'])
        except Exception:
            used_images = 0

        for file in uploaded_files:
            file_bytes_raw = file.read()
            if is_video_file(getattr(file, 'name', ''), getattr(file, 'type', '')):
                # 使用系统临时文件进行抽帧，文件会在关闭后自动删除，不落盘到项目目录
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(getattr(file, 'name', 'video.mp4'))[1] or '.mp4', delete=True) as tf:
                    tf.write(file_bytes_raw)
                    tf.flush()
                    
                    # 获取视频基本信息
                    cap = cv2.VideoCapture(tf.name)
                    if cap.isOpened():
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        duration = total_frames / fps if fps > 0 else 0
                        cap.release()
                    else:
                        total_frames = 1
                        fps = 1
                        duration = 0
                    
                    # 完全动态的关键帧抽取：不受图像配额限制
                    frames = extract_video_frames_to_pil(tf.name, max_frames=None)
                    
                    # 显示抽取信息
                    if len(frames) == total_frames:
                        st.info(f"🎬 Short video detected: Extracted all {total_frames} frames (Duration: {duration:.1f}s)")
                    else:
                        st.info(f"🎬 Smart keyframe extraction: Selected {len(frames)} keyframes from {total_frames} frames (Duration: {duration:.1f}s)")
                        
                images.extend(frames)
                persist_flags.extend([False] * len(frames))
                # 不记录视频文件路径，完全在临时文件中处理
            else:
                file_bytes = np.asarray(bytearray(file_bytes_raw), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                images.append(img)
                persist_flags.append(True)
        # 只显示上传的文件，不显示YouTube视频帧预览
        display_images = []
        display_persist_flags = []
        
        # 分离上传文件和YouTube视频帧
        for i, (image, to_persist) in enumerate(zip(images, persist_flags)):
            if to_persist:  # 只显示需要持久化的图片（上传的文件）
                display_images.append(image)
                display_persist_flags.append(to_persist)
        
        if display_images:
            with upload_image_preview.container():
                Library(display_images)
        
        # 仅持久化普通上传图片；视频帧不落盘
        for image, to_persist in zip(images, persist_flags):
            if not to_persist:
                continue
            t = datetime.datetime.now()
            img_hash = hashlib.md5(image.tobytes()).hexdigest()
            filename = os.path.join(LOGDIR, 'serve_images', f'{t.year}-{t.month:02d}-{t.day:02d}', f'{img_hash}.jpg')
            filenames.append(filename)
            if not os.path.isfile(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                image.save(filename)
    return images, filenames


def get_selected_worker_ip():
    if use_chatgpt:
        return chatgpt_worker_url
    
    ret = requests.post(controller_url + '/get_worker_address',
            json={'model': selected_model})
    worker_addr = ret.json()['address']
    return worker_addr


def save_chat_history():
    messages = st.session_state.messages
    new_messages = []
    for message in messages:
        new_message = {'role': message['role'], 'content': message['content']}
        if 'filenames' in message:
            new_message['filenames'] = message['filenames']
        new_messages.append(new_message)
    if len(new_messages) > 0:
        fout = open(get_conv_log_filename(), 'a')
        data = {
            'type': 'chat',
            'model': selected_model,
            'messages': new_messages,
        }
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')
        fout.close()


def generate_response(messages):
    send_messages = [{'role': 'system', 'content': system_message_default + '\n\n' + persona_rec}]
    for message in messages:
        if message['role'] == 'user':
            user_message = {'role': 'user', 'content': message['content']}
            if 'image' in message and len('image') > 0:
                user_message['image'] = []
                for image in message['image']:
                    user_message['image'].append(pil_image_to_base64(image))
            send_messages.append(user_message)
        else:
            send_messages.append({'role': 'assistant', 'content': message['content']})
    pload = {
        'model': selected_model,
        'prompt': send_messages,
        'temperature': float(temperature),
        'top_p': float(top_p),
        'max_new_tokens': max_length,
        'max_input_tiles': max_input_tiles,
        'repetition_penalty': float(repetition_penalty),
    }
    worker_addr = get_selected_worker_ip()
    headers = {'User-Agent': 'InternVL-Chat Client'}
    placeholder, output = st.empty(), ''
    try:
        response = requests.post(worker_addr + '/worker_generate_stream',
                                 headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=True, delimiter=b'\0'):
            if chunk:
                data = json.loads(chunk.decode())
                if data['error_code'] == 0:
                    output = data['text']
                    # Phi3-3.8B will produce abnormal `�` output
                    if '4B' in selected_model and '�' in output[-2:]:
                        output = output.replace('�', '')
                        break
                    placeholder.markdown(output + '▌')
                else:
                    output = data['text'] + f" (error_code: {data['error_code']})"
                    placeholder.markdown(output)
        if ('\[' in output and '\]' in output) or ('\(' in output and '\)' in output):
            output = output.replace('\[', '$').replace('\]', '$').replace('\(', '$').replace('\)', '$')
        placeholder.markdown(output)
    except requests.exceptions.RequestException as e:
        placeholder.markdown(server_error_msg)
    return output


def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def clear_chat_history():
    st.session_state.messages = []
    st.session_state['image_select'] = -1
    # 清除视频帧和字幕
    if 'video_frames' in st.session_state:
        st.session_state.video_frames = []
    if 'video_url' in st.session_state:
        st.session_state.video_url = ''
    if 'video_subtitles' in st.session_state:
        del st.session_state.video_subtitles
    if 'video_title' in st.session_state:
        del st.session_state.video_title


def clear_file_uploader():
    st.session_state.uploader_key += 1
    st.rerun()


def combined_func(func_list):
    for func in func_list:
        func()


def show_one_or_multiple_images(message, total_image_num, is_input=True):
    if 'image' in message:
        if is_input:
            total_image_num = total_image_num + len(message['image'])
            
            # 检查是否有视频帧
            video_frames_count = 0
            regular_images_count = 0
            
            if 'video_frames' in st.session_state and st.session_state.video_frames:
                video_frames_count = len(st.session_state.video_frames)
                regular_images_count = len(message['image']) - video_frames_count
            else:
                regular_images_count = len(message['image'])
            
            if lan == 'English':
                if video_frames_count > 0 and regular_images_count > 0:
                    label = f"(In this conversation, {regular_images_count} image(s) uploaded, {video_frames_count} frames from video processed, {total_image_num} total)"
                elif video_frames_count > 0:
                    label = f"(In this conversation, {video_frames_count} frames from video processed, {total_image_num} total)"
                elif regular_images_count == 1 and total_image_num == 1:
                    label = f"(In this conversation, {regular_images_count} image was uploaded, {total_image_num} image in total)"
                elif regular_images_count == 1 and total_image_num > 1:
                    label = f"(In this conversation, {regular_images_count} image was uploaded, {total_image_num} images in total)"
                else:
                    label = f"(In this conversation, {regular_images_count} images were uploaded, {total_image_num} images in total)"
            else:
                if video_frames_count > 0 and regular_images_count > 0:
                    label = f"(在本次对话中，上传了{regular_images_count}张图片，处理了{video_frames_count}帧视频，总共{total_image_num}张)"
                elif video_frames_count > 0:
                    label = f"(在本次对话中，处理了{video_frames_count}帧视频，总共{total_image_num}张)"
                else:
                    label = f"(在本次对话中，上传了{regular_images_count}张图片，总共上传了{total_image_num}张图片)"
        
        # 显示聊天记录中的图片（现在只包含用户上传的图片）
        if message['image']:
            upload_image_preview = st.empty()
            with upload_image_preview.container():
                Library(message['image'])
        
        # 如果有视频帧被处理，显示提示信息
        if 'video_frames' in st.session_state and st.session_state.video_frames:
            video_frames_count = len(st.session_state.video_frames)
            subtitle_info = ""
            if 'video_subtitles' in st.session_state and st.session_state.video_subtitles:
                subtitle_count = len(st.session_state.video_subtitles)
                if lan == 'English':
                    subtitle_info = f" and {subtitle_count} subtitle file(s)"
                else:
                    subtitle_info = f" 和 {subtitle_count} 个字幕文件"
            
            if lan == 'English':
                st.info(f"🎥 Video frames ({video_frames_count} frames){subtitle_info} are being processed in the background")
            else:
                st.info(f"🎥 视频帧（{video_frames_count}帧）{subtitle_info}正在后台处理中")
        
        # 只在有上传的图片时显示标签，纯视频处理时不显示
        if is_input and regular_images_count > 0:
            st.markdown(label)


def find_bounding_boxes(response):
    pattern = re.compile(r'<ref>\s*(.*?)\s*</ref>\s*<box>\s*(\[\[.*?\]\])\s*</box>')
    matches = pattern.findall(response)
    results = []
    for match in matches:
        results.append((match[0], eval(match[1])))
    returned_image = None
    for message in st.session_state.messages:
        if message['role'] == 'user' and 'image' in message and len(message['image']) > 0:
            last_image = message['image'][-1]
            width, height = last_image.size
            returned_image = last_image.copy()
            draw = ImageDraw.Draw(returned_image)
    for result in results:
        line_width = max(1, int(min(width, height) / 200))
        random_color = (random.randint(0, 128), random.randint(0, 128), random.randint(0, 128))
        category_name, coordinates = result
        coordinates = [(float(x[0]) / 1000, float(x[1]) / 1000, float(x[2]) / 1000, float(x[3]) / 1000) for x in coordinates]
        coordinates = [(int(x[0] * width), int(x[1] * height), int(x[2] * width), int(x[3] * height)) for x in coordinates]
        for box in coordinates:
            draw.rectangle(box, outline=random_color, width=line_width)
            font = ImageFont.truetype('static/SimHei.ttf', int(20 * line_width / 2))
            text_size = font.getbbox(category_name)
            text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]
            text_position = (box[0], max(0, box[1] - text_height))
            draw.rectangle(
                [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
                fill=random_color
            )
            draw.text(text_position, category_name, fill='white', font=font)
    return returned_image if len(matches) > 0 else None


def query_image_generation(response, sd_worker_url, timeout=15):
    sd_worker_url = f'{sd_worker_url}/generate_image/'
    pattern = r'```drawing-instruction\n(.*?)\n```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        payload = {'caption': match.group(1)}
        print('drawing-instruction:', payload)
        response = requests.post(sd_worker_url, json=payload, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    else:
        return None


def regenerate():
    st.session_state.messages = st.session_state.messages[:-1]
    st.rerun()


logo_code = """
<svg width="600" height="120" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color: #1e3a8a; stop-opacity: 1" />
      <stop offset="100%" style="stop-color: #3b82f6; stop-opacity: 1" />
    </linearGradient>
    <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color: #f97316; stop-opacity: 1" />
      <stop offset="100%" style="stop-color: #fbbf24; stop-opacity: 1" />
    </linearGradient>
  </defs>
  <!-- Curved swoosh element -->
  <path d="M 20 50 Q 200 15 300 25 Q 400 35 580 50" stroke="url(#gradient2)" stroke-width="4" fill="none" />
  <!-- Main brand text -->
  <text x="150" y="80" font-size="60" font-weight="bold" fill="url(#gradient1)" style="font-family: Arial, sans-serif; font-style: italic;">
    PacDent
  </text>
  <!-- Registered trademark symbol -->
  <text x="300" y="60" font-size="20" fill="url(#gradient1)" style="font-family: Arial, sans-serif;">®</text>
  <!-- Tagline -->
  <text x="150" y="105" font-size="20" fill="#6b7280" style="font-family: Arial, sans-serif;">
    Passion for Excellence
  </text>
</svg>
"""

# App title
st.set_page_config(
    page_title='Pac-Dent MediaMind',
    page_icon='static/pac-dent-logo.png'
)

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

system_message_default = '我是 Pac-Dent MediaMind，一个专业的AI媒体分析和理解平台，专注于提供高质量的媒体内容分析和智能处理服务。'

system_message_editable = '请尽可能详细地回答用户的问题。'

# Replicate Credentials
with st.sidebar:
    model_list = get_model_list()
    # "[![Open in GitHub](https://github.com/codespaces/badge.svg)](https://github.com/OpenGVLab/InternVL)"
    lan = st.selectbox('#### Language / 语言', ['English', '中文'], on_change=st.rerun,
                       help='This is only for switching the UI language. 这仅用于切换UI界面的语言。')
    if lan == 'English':
        # st.logo(logo_code, link='https://github.com/OpenGVLab/InternVL', icon_image=logo_code)
        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Choose a Pac-Dent MediaMind model', model_list, key='selected_model',
                                              on_change=clear_chat_history,
                                              help='Select the AI model for media analysis and understanding.')
        with st.expander('🤖 System Prompt'):
            persona_rec = st.text_area('System Prompt', value=system_message_editable,
                                       help='System prompt is a pre-defined message used to instruct the assistant at the beginning of a conversation.',
                                       height=200)
        with st.expander('🔥 Advanced Options'):
            temperature = st.slider('temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            top_p = st.slider('top_p', min_value=0.0, max_value=1.0, value=0.95, step=0.05)
            repetition_penalty = st.slider('repetition_penalty', min_value=1.0, max_value=1.5, value=1.1, step=0.02)
            max_length = st.slider('max_new_token', min_value=0, max_value=4096, value=1024, step=128)
            max_input_tiles = st.slider('max_input_tiles (control image resolution)', min_value=1, max_value=24,
                                        value=12, step=1)
            st.info('🎥 Video frame extraction strategy: The system will automatically determine the number of frames to extract based on video length and content')
            st.caption('• Short videos (≤50 frames): Extract all frames\n• Medium videos: Smart extraction based on quota\n• Long videos: Uniform sampling to maintain representativeness')
        # 视频链接输入
        st.subheader('🎥 Enter a video link')
        # 初始化session state
        if 'video_url' not in st.session_state:
            st.session_state.video_url = ''
        
        video_url = st.text_input('Video Link', 
                                 value=st.session_state.video_url,
                                 placeholder='Paste your Video Link here',
                                 help='Enter a video link, then click the button to process the video', 
                                 key='video_url_input',
                                 label_visibility="visible")
        
        # 更新session state
        st.session_state.video_url = video_url
        
        # 视频处理按钮和清空按钮并排
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button('🎬 Process Video', type='primary'):
                if video_url and video_url.strip():
                    if is_video_url(video_url.strip()):
                        with st.spinner('Processing video...'):
                            result = stream_video_frames(video_url.strip())
                            if result and isinstance(result, dict) and result['frames']:
                                # 将视频帧添加到session state
                                if 'video_frames' not in st.session_state:
                                    st.session_state.video_frames = []
                                st.session_state.video_frames.extend(result['frames'])
                                
                                # 存储字幕信息
                                if result['subtitles']:
                                    st.session_state.video_subtitles = result['subtitles']
                                    st.session_state.video_title = result['title']
                                
                                subtitle_info = f" and {len(result['subtitles'])} subtitle file(s)" if result['subtitles'] else ""
                                st.success(f"Successfully processed video, extracted {len(result['frames'])} frames{subtitle_info}")
                                
                                # 显示字幕预览
                                if result['subtitles']:
                                    with st.expander("📝 Subtitle Preview", expanded=True):
                                        for i, subtitle in enumerate(result['subtitles']):
                                            st.write(f"**Subtitle {i+1}:**")
                                            # 显示完整字幕内容
                                            st.text_area(f"Full Subtitle {i+1}:", subtitle['content'], height=150, key=f"preview_full_{i}")
                                            
                                            # Text Only部分已移除
                            else:
                                st.error("Failed to process video")
                    else:
                        st.warning("Please enter a valid video link")
                else:
                    st.warning("Please enter a video link")
        
        with col2:
            # 只要有视频URL输入就显示清空按钮，允许随时取消处理
            if video_url and video_url.strip():
                if st.button('🗑️ Clear', help='Clear video URL and stop processing'):
                    st.session_state.video_frames = []
                    st.session_state.video_url = ''
                    if 'video_subtitles' in st.session_state:
                        del st.session_state.video_subtitles
                    if 'video_title' in st.session_state:
                        del st.session_state.video_title
        
        upload_image_preview = st.empty()
        uploaded_files = st.file_uploader('Upload files', accept_multiple_files=True,
                                          type=['png', 'jpg', 'jpeg', 'webp', 'mp4', 'mov', 'avi', 'mkv', 'webm'],
                                          help=f'你可以上传多张图像（最多{max_image_limit}张）或者一个视频。',
                                          key=f'uploader_{st.session_state.uploader_key}',
                                          on_change=st.rerun)
        uploaded_pil_images, save_filenames = load_upload_file_and_show()
        
        # 字幕信息已隐藏 - 不再显示给用户
        # if 'video_subtitles' in st.session_state and st.session_state.video_subtitles:
        #     with st.expander('📝 Video Subtitles', expanded=True):
        #         for i, subtitle in enumerate(st.session_state.video_subtitles):
        #             st.write(f"**Subtitle File {i+1}:**")
        #             # 显示完整的字幕内容（包括时间戳）
        #             st.text_area(f"Full Content {i+1}:", subtitle['content'], height=200, key=f"full_subtitle_{i}")
        #             
        #             # 提取并显示纯文本版本
        #             lines = subtitle['content'].split('\n')
        #             text_lines = [line for line in lines if line and not line.startswith('WEBVTT') and not '-->' in line and not line.isdigit()]
        #             subtitle_text = " ".join(text_lines)
        #             if subtitle_text.strip():
        #                 st.write("**Text Only:**")
        #                 st.text_area(f"Text Content {i+1}:", subtitle_text.strip(), height=100, key=f"text_subtitle_{i}")
        #             else:
        #                 st.write("No text content found in this subtitle file.")
    else:
        st.subheader('模型和参数')
        selected_model = st.sidebar.selectbox('选择一个 Pac-Dent MediaMind 模型', model_list, key='selected_model',
                                              on_change=clear_chat_history,
                                              help='选择用于媒体分析和理解的AI模型。')
        with st.expander('🤖 系统提示'):
            persona_rec = st.text_area('系统提示', value=system_message_editable,
                                       help='系统提示是在对话开始时用于指示助手的预定义消息。',
                                       height=200)
        with st.expander('🔥 高级选项'):
            temperature = st.slider('temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            top_p = st.slider('top_p', min_value=0.0, max_value=1.0, value=0.95, step=0.05)
            repetition_penalty = st.slider('重复惩罚', min_value=1.0, max_value=1.5, value=1.1, step=0.02)
            max_length = st.slider('最大输出长度', min_value=0, max_value=4096, value=1024, step=128)
            max_input_tiles = st.slider('最大图像块数 (控制图像分辨率)', min_value=1, max_value=24, value=12, step=1)
            st.info('🎥 Video frame extraction strategy: The system will automatically determine the number of frames to extract based on video length and content')
            st.caption('• Short videos (≤50 frames): Extract all frames\n• Medium videos: Smart extraction based on quota\n• Long videos: Uniform sampling to maintain representativeness')
        
        # 视频链接输入
        st.subheader('🎥 或输入视频链接')
        # 初始化session state
        if 'video_url' not in st.session_state:
            st.session_state.video_url = ''
        
        video_url = st.text_input('视频链接', 
                                 value=st.session_state.video_url,
                                 placeholder='https://www.youtube.com/watch?v=... 或 https://vimeo.com/... 或直接视频文件链接',
                                 help='输入视频链接，然后点击按钮处理视频', 
                                 key='video_url_input',
                                 label_visibility="visible")
        
        # 更新session state
        st.session_state.video_url = video_url
        
        # 视频处理按钮和清空按钮并排
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button('🎬 处理视频', type='primary'):
                if video_url and video_url.strip():
                    if is_video_url(video_url.strip()):
                        with st.spinner('Processing video...'):
                            result = stream_video_frames(video_url.strip())
                            if result and isinstance(result, dict) and result['frames']:
                                # 将视频帧添加到session state
                                if 'video_frames' not in st.session_state:
                                    st.session_state.video_frames = []
                                st.session_state.video_frames.extend(result['frames'])
                                
                                # 存储字幕信息
                                if result['subtitles']:
                                    st.session_state.video_subtitles = result['subtitles']
                                    st.session_state.video_title = result['title']
                                
                                subtitle_info = f" 和 {len(result['subtitles'])} 个字幕文件" if result['subtitles'] else ""
                                st.success(f"成功处理视频，提取了 {len(result['frames'])} 帧{subtitle_info}")
                                
                                # 显示字幕预览
                                if result['subtitles']:
                                    with st.expander("📝 字幕预览", expanded=True):
                                        for i, subtitle in enumerate(result['subtitles']):
                                            st.write(f"**字幕 {i+1}:**")
                                            # 显示完整字幕内容
                                            st.text_area(f"完整字幕 {i+1}:", subtitle['content'], height=150, key=f"preview_full_{i}")
                                            
                                            # 纯文本版本部分已移除
                            else:
                                st.error("处理视频失败")
                    else:
                        st.warning("Please enter a valid video link")
                else:
                    st.warning("Please enter a video link")
        
        with col2:
            # 只要有视频URL输入就显示清空按钮，允许随时取消处理
            if video_url and video_url.strip():
                if st.button('🗑️ 清空', help='清空视频链接并停止处理'):
                    st.session_state.video_frames = []
                    st.session_state.video_url = ''
                    if 'video_subtitles' in st.session_state:
                        del st.session_state.video_subtitles
                    if 'video_title' in st.session_state:
                        del st.session_state.video_title
        
        upload_image_preview = st.empty()
        uploaded_files = st.file_uploader('上传文件', accept_multiple_files=True,
                                          type=['png', 'jpg', 'jpeg', 'webp', 'mp4', 'mov', 'avi', 'mkv', 'webm'],
                                          help=f'你可以上传多张图像（最多{max_image_limit}张）或者一个视频。',
                                          key=f'uploader_{st.session_state.uploader_key}',
                                          on_change=st.rerun)
        uploaded_pil_images, save_filenames = load_upload_file_and_show()
        
        # 字幕信息已隐藏 - 不再显示给用户
        # if 'video_subtitles' in st.session_state and st.session_state.video_subtitles:
        #     with st.expander('📝 视频字幕', expanded=True):
        #         for i, subtitle in enumerate(st.session_state.video_subtitles):
        #             st.write(f"**字幕文件 {i+1}:**")
        #             # 显示完整的字幕内容（包括时间戳）
        #             st.text_area(f"完整内容 {i+1}:", subtitle['content'], height=200, key=f"full_subtitle_{i}")
        #             
        #             # 提取并显示纯文本版本
        #             lines = subtitle['content'].split('\n')
        #             text_lines = [line for line in lines if line and not line.startswith('WEBVTT') and not '-->' in line and not line.isdigit()]
        #             subtitle_text = " ".join(text_lines)
        #             if subtitle_text.strip():
        #                 st.write("**纯文本版本:**")
        #                 st.text_area(f"文本内容 {i+1}:", subtitle_text.strip(), height=100, key=f"text_subtitle_{i}")
        #             else:
        #                 st.write("此字幕文件中未找到文本内容。")

# Logo styling
st.markdown("""
<style>
.logo-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

if lan == 'English':
    st.markdown('<div class="logo-container">' + logo_code + '</div>', unsafe_allow_html=True)
    st.caption('An AI-Powered Media Analysis and Understanding Platform')
else:
    st.markdown('<div class="logo-container">' + logo_code + '</div>', unsafe_allow_html=True)
    st.caption('AI驱动的媒体分析和理解平台')

# Store LLM generated responses
if 'messages' not in st.session_state.keys():
    clear_chat_history()

gallery_placeholder = st.empty()
with gallery_placeholder.container():
    examples = ['gallery/prod_9.jpg', 'gallery/astro_on_unicorn.png',
                'gallery/prod_12.png', 'gallery/prod_en_17.png',
                'gallery/prod_4.png', 'gallery/cheetah.png', 'gallery/prod_1.jpeg']
    images = [Image.open(image) for image in examples]
    if lan == 'English':
        captions = ["What's at the far end of the image?",
                    'Could you help me draw a picture like this one?',
                    'What are the consequences of the easy decisions shown in this image?',
                    "I'm on a diet, but I really want to eat them.",
                    'Is this a real plant? Analyze the reasons.',
                    'Detect the <ref>the middle leopard</ref> in the image with its bounding box.',
                    'Please identify and label all objects in the following image.']
    else:
        captions = ['画面最远处是什么?',
                    '请画一张类似这样的画',
                    '这张图上 easy decisions 导致了什么后果?',
                    '我在减肥，但我真的很想吃这个。',
                    '这是真的植物吗？分析原因',
                    '在以下图像中进行目标检测，并标出所有物体。',
                    '这幅图的氛围如何？']
    img_idx = image_select(
        label='',
        images=images,
        captions=captions,
        use_container_width=True,
        index=-1,
        return_value='index',
        key='image_select'
    )
    # if lan == 'English':
        # st.caption(
        #     'Note: For non-commercial research use only. AI responses may contain errors. Users should not spread or allow others to spread hate speech, violence, pornography, or fraud-related harmful information.')
    # else:
        # st.caption('注意：仅限非商业研究使用。用户应不传播或允许他人传播仇恨言论、暴力、色情内容或与欺诈相关的有害信息。')
    if img_idx != -1 and len(st.session_state.messages) == 0 and selected_model is not None:
        gallery_placeholder.empty()
        st.session_state.messages.append({'role': 'user', 'content': captions[img_idx], 'image': [images[img_idx]],
                                          'filenames': [examples[img_idx]]})
        st.rerun()  # Fixed an issue where examples were not emptied

if len(st.session_state.messages) > 0:
    gallery_placeholder.empty()

# Display or clear chat messages
total_image_num = 0
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        show_one_or_multiple_images(message, total_image_num, is_input=message['role'] == 'user')
        if 'image' in message and message['role'] == 'user':
            total_image_num += len(message['image'])

input_disable_flag = (len(model_list) == 0) or total_image_num + len(uploaded_files) > max_image_limit
if lan == 'English':
    st.sidebar.button('Clear Chat History',
                      on_click=partial(combined_func, func_list=[clear_chat_history, clear_file_uploader]))
    if input_disable_flag:
        prompt = st.chat_input('Too many images have been uploaded. Please clear the history.',
                               disabled=input_disable_flag)
    else:
        prompt = st.chat_input('Send messages to Pac-Dent MediaMind', disabled=input_disable_flag)
else:
    st.sidebar.button('清空聊天记录', on_click=partial(combined_func, func_list=[clear_chat_history, clear_file_uploader]))
    if input_disable_flag:
        prompt = st.chat_input('输入的图片太多了，请清空历史记录。', disabled=input_disable_flag)
    else:
        prompt = st.chat_input('给 "Pac-Dent MediaMind" 发送消息', disabled=input_disable_flag)

alias_instructions = {
    '目标检测': '在以下图像中进行目标检测，并标出所有物体。',
    '检测': '在以下图像中进行目标检测，并标出所有物体。',
    'object detection': 'Please identify and label all objects in the following image.',
    'detection': 'Please identify and label all objects in the following image.'
}

if prompt:
    prompt = alias_instructions[prompt] if prompt in alias_instructions else prompt
    gallery_placeholder.empty()
    image_list = uploaded_pil_images
    
    # 将视频帧添加到发送给AI的图像列表中，但不显示在聊天记录中
    all_images_for_ai = image_list.copy()
    if 'video_frames' in st.session_state and st.session_state.video_frames:
        all_images_for_ai.extend(st.session_state.video_frames)
    
    # 如果有字幕，将字幕信息添加到提示中
    enhanced_prompt = prompt
    if 'video_subtitles' in st.session_state and st.session_state.video_subtitles:
        subtitle_text = ""
        for subtitle in st.session_state.video_subtitles:
            # 简单提取字幕文本（去除时间戳）
            lines = subtitle['content'].split('\n')
            text_lines = [line for line in lines if line and not line.startswith('WEBVTT') and not '-->' in line and not line.isdigit()]
            subtitle_text += " ".join(text_lines) + "\n"
        
        if subtitle_text.strip():
            enhanced_prompt = f"{prompt}\n\nVideo subtitles for context:\n{subtitle_text.strip()}"
    
    # 聊天记录中只保存用户上传的图片
    st.session_state.messages.append(
        {'role': 'user', 'content': prompt, 'image': image_list, 'filenames': save_filenames})
    
    with st.chat_message('user'):
        st.write(prompt)
        show_one_or_multiple_images(st.session_state.messages[-1], total_image_num, is_input=True)
    if image_list:
        clear_file_uploader()

# Generate a new response if last message is not from assistant
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]['role'] != 'assistant':
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            if not prompt:
                prompt = st.session_state.messages[-1]['content']
            
            # 临时修改最后一条用户消息，添加视频帧和字幕用于AI处理
            messages_for_ai = st.session_state.messages.copy()
            if 'video_frames' in st.session_state and st.session_state.video_frames:
                last_user_message = messages_for_ai[-1]
                if 'image' in last_user_message:
                    # 创建包含视频帧的图像列表副本
                    all_images = last_user_message['image'].copy()
                    all_images.extend(st.session_state.video_frames)
                    last_user_message = last_user_message.copy()
                    last_user_message['image'] = all_images
                    
                    # 如果有字幕，使用增强的提示
                    if 'video_subtitles' in st.session_state and st.session_state.video_subtitles:
                        subtitle_text = ""
                        for subtitle in st.session_state.video_subtitles:
                            lines = subtitle['content'].split('\n')
                            text_lines = [line for line in lines if line and not line.startswith('WEBVTT') and not '-->' in line and not line.isdigit()]
                            subtitle_text += " ".join(text_lines) + "\n"
                        
                        if subtitle_text.strip():
                            last_user_message['content'] = f"{last_user_message['content']}\n\nVideo subtitles for context:\n{subtitle_text.strip()}"
                    
                    messages_for_ai[-1] = last_user_message
            
            response = generate_response(messages_for_ai)
            message = {'role': 'assistant', 'content': response}
        with st.spinner('Drawing...'):
            if '<ref>' in response:
                has_returned_image = find_bounding_boxes(response)
                message['image'] = [has_returned_image] if has_returned_image else []
            if '```drawing-instruction' in response:
                has_returned_image = query_image_generation(response, sd_worker_url=sd_worker_url)
                message['image'] = [has_returned_image] if has_returned_image else []
            st.session_state.messages.append(message)
            show_one_or_multiple_images(message, total_image_num, is_input=False)

if len(st.session_state.messages) > 0:
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1.3])
    text1 = 'Clear Chat History' if lan == 'English' else '清空聊天记录'
    text2 = 'Regenerate' if lan == 'English' else '重新生成'
    text3 = 'Copy' if lan == 'English' else '复制回答'
    with col1:
        st.button(text1, on_click=partial(combined_func, func_list=[clear_chat_history, clear_file_uploader]),
                  key='clear_chat_history_button')
    with col2:
        st.button(text2, on_click=regenerate, key='regenerate_button')

print(st.session_state.messages)
save_chat_history()
