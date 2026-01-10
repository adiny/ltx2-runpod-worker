VERSION = "4.6.0-DIRECT-DOWNLOAD"

import os
import sys
import json
import requests
import torch
import runpod
import base64
import tempfile
import time
import subprocess
import soundfile as sf
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Use overlay filesystem which should have more space
MODEL_PATH = "/root/LTX-2"
os.environ["TMPDIR"] = "/root/tmp"
os.environ["HF_HOME"] = "/root/hf_cache"

pipe = None
HF_REPO = "Lightricks/LTX-2"
HF_API = "https://huggingface.co/api/models"


def download_file(url, dest_path, desc=""):
    """Download a single file with progress"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return dest_path


def download_model_direct():
    """Download model files directly from Hugging Face"""
    print(f"📥 Downloading {HF_REPO} directly...")
    
    # Get file list from HF API
    api_url = f"{HF_API}/{HF_REPO}"
    response = requests.get(api_url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to get model info: {response.status_code}")
    
    # Get siblings (files)
    tree_url = f"https://huggingface.co/api/models/{HF_REPO}/tree/main"
    
    def get_all_files(path=""):
        url = f"https://huggingface.co/api/models/{HF_REPO}/tree/main/{path}" if path else tree_url
        resp = requests.get(url)
        if resp.status_code != 200:
            return []
        
        files = []
        for item in resp.json():
            if item['type'] == 'file':
                if not item['path'].endswith('.md') and '.git' not in item['path']:
                    files.append(item['path'])
            elif item['type'] == 'directory':
                files.extend(get_all_files(item['path']))
        return files
    
    files = get_all_files()
    print(f"   Found {len(files)} files to download")
    
    # Download each file
    base_url = f"https://huggingface.co/{HF_REPO}/resolve/main"
    
    for i, file_path in enumerate(files):
        dest = os.path.join(MODEL_PATH, file_path)
        
        if os.path.exists(dest):
            continue
            
        url = f"{base_url}/{file_path}"
        print(f"   [{i+1}/{len(files)}] {file_path}")
        
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
    
    print("✅ Download complete!")


def get_audio_duration(file_path):
    f = sf.SoundFile(file_path)
    return len(f) / f.samplerate


def download_audio(url, save_path):
    print(f"📥 Downloading audio from: {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download audio. Status: {response.status_code}")


def save_final_output(video_frames, input_audio_path, generated_audio_waveform, output_path, fps=24):
    from diffusers.utils import export_to_video
    
    temp_video = output_path.replace(".mp4", "_visual.mp4")
    temp_audio = None
    
    export_to_video(video_frames, temp_video, fps=fps)
    
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_video]
    
    if input_audio_path and os.path.exists(input_audio_path):
        print("🎵 Merging USER audio...")
        ffmpeg_cmd.extend(["-i", input_audio_path, "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
    elif generated_audio_waveform is not None:
        print("🎵 Merging GENERATED audio...")
        temp_audio = output_path.replace(".mp4", ".wav")
        if torch.is_tensor(generated_audio_waveform):
            generated_audio_waveform = generated_audio_waveform.cpu().float().numpy()
        if generated_audio_waveform.ndim > 1 and generated_audio_waveform.shape[0] < generated_audio_waveform.shape[1]:
            generated_audio_waveform = generated_audio_waveform.T
        sf.write(temp_audio, generated_audio_waveform, 24000)
        ffmpeg_cmd.extend(["-i", temp_audio, "-c:v", "copy", "-c:a", "aac", "-shortest"])
    else:
        print("🔇 No audio source.")
        os.rename(temp_video, output_path)
        return

    ffmpeg_cmd.extend(["-strict", "experimental", output_path])
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(temp_video):
        os.remove(temp_video)
    if temp_audio and os.path.exists(temp_audio):
        os.remove(temp_audio)


def load_model():
    global pipe
    if pipe is not None:
        return pipe
    
    print(f"🚀 Version: {VERSION}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check disk space on all paths
    print("💾 Disk space:")
    for path in ["/", "/tmp", "/root", "/workspace"]:
        try:
            total, used, free = shutil.disk_usage(path)
            print(f"   {path}: {free // (1024**3)}GB free / {total // (1024**3)}GB total")
        except:
            print(f"   {path}: N/A")
    
    os.makedirs("/root/tmp", exist_ok=True)
    os.makedirs("/root/hf_cache", exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    config_path = os.path.join(MODEL_PATH, "config.json")
    if not os.path.exists(config_path):
        download_model_direct()
    else:
        print(f"✅ Model cached at {MODEL_PATH}")
    
    print("⏳ Loading pipeline...")
    from diffusers import LTXPipeline
    pipe = LTXPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True
    )
    pipe.enable_model_cpu_offload()
    print("✅ Ready!")
    return pipe


def handler(event):
    temp_files = []
    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt")
        audio_url = job_input.get("audio_url")
        
        if not prompt:
            return {"error": "Missing prompt"}
        
        width = job_input.get("width", 768)
        height = job_input.get("height", 512)
        fps = job_input.get("fps", 24)
        num_frames = job_input.get("num_frames", 121)
        input_audio_path = None
        
        if audio_url:
            input_audio_path = tempfile.mktemp(suffix=".mp3", dir="/root/tmp")
            temp_files.append(input_audio_path)
            download_audio(audio_url, input_audio_path)
            
            duration = get_audio_duration(input_audio_path)
            calculated_frames = int(duration * fps) + 8
            calculated_frames = calculated_frames - (calculated_frames % 8) + 1
            
            print(f"🎙️ Audio: {duration:.2f}s -> {calculated_frames} frames")
            num_frames = min(calculated_frames, 257)

        pipeline = load_model()
        
        print(f"🎬 Generating: {prompt[:50]}...")
        print(f"   {width}x{height}, {num_frames} frames")
        
        start = time.time()
        output = pipeline(
            prompt=prompt,
            negative_prompt=job_input.get("negative_prompt", "low quality"),
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=job_input.get("num_inference_steps", 30),
            output_type="pt"
        )
        gen_time = time.time() - start
        
        print(f"✅ Done in {gen_time:.1f}s")
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/root/tmp") as f:
            out_path = f.name
            temp_files.append(out_path)
            
        generated_audio = output.audio[0] if hasattr(output, 'audio') and output.audio is not None else None
        save_final_output(output.frames[0], input_audio_path, generated_audio, out_path, fps)
        
        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        
        torch.cuda.empty_cache()
        
        return {
            "video_base64": video_b64,
            "generation_time": round(gen_time, 2),
            "frames": num_frames,
            "has_audio": input_audio_path is not None or generated_audio is not None
        }

    except Exception as e:
        import traceback
        print(f"❌ {e}\n{traceback.format_exc()}")
        return {"error": str(e), "traceback": traceback.format_exc()}
    finally:
        for f in temp_files:
            if f and os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
