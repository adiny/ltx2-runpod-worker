VERSION = "5.3.0-LTX2-HFHUB"

import os
import sys
import torch
import runpod
import base64
import tempfile
import time
import subprocess
import soundfile as sf
import requests
import shutil

# Disable XET to avoid disk quota issues
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TMPDIR"] = "/root/tmp"
os.environ["HF_HOME"] = "/root/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/hf_cache"

MODEL_PATH = "/root/LTX-2"
pipe = None
HF_REPO = "Lightricks/LTX-2"


def download_model():
    """Download model using huggingface_hub"""
    print(f"📥 Downloading {HF_REPO} using huggingface_hub...")
    
    os.makedirs("/root/tmp", exist_ok=True)
    os.makedirs("/root/hf_cache", exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    from huggingface_hub import snapshot_download
    
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=MODEL_PATH,
        cache_dir="/root/hf_cache",
        ignore_patterns=["*.md", "*.git*", "*.mp4"],
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
    )
    
    print("✅ Download complete!")


def get_audio_duration(file_path):
    f = sf.SoundFile(file_path)
    return len(f) / f.samplerate


def download_audio(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)


def save_final_output(video_frames, input_audio_path, generated_audio_waveform, output_path, fps=24):
    from diffusers.utils import export_to_video
    
    temp_video = output_path.replace(".mp4", "_visual.mp4")
    temp_audio = None
    
    export_to_video(video_frames, temp_video, fps=fps)
    
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_video]
    
    if input_audio_path and os.path.exists(input_audio_path):
        ffmpeg_cmd.extend(["-i", input_audio_path, "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
    elif generated_audio_waveform is not None:
        temp_audio = output_path.replace(".mp4", ".wav")
        if torch.is_tensor(generated_audio_waveform):
            generated_audio_waveform = generated_audio_waveform.cpu().float().numpy()
        if generated_audio_waveform.ndim > 1 and generated_audio_waveform.shape[0] < generated_audio_waveform.shape[1]:
            generated_audio_waveform = generated_audio_waveform.T
        sf.write(temp_audio, generated_audio_waveform, 24000)
        ffmpeg_cmd.extend(["-i", temp_audio, "-c:v", "copy", "-c:a", "aac", "-shortest"])
    else:
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
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    print("💾 Disk space:")
    for path in ["/", "/root"]:
        try:
            total, used, free = shutil.disk_usage(path)
            print(f"   {path}: {free // (1024**3)}GB free")
        except:
            pass
    
    # Download model if needed
    config_path = os.path.join(MODEL_PATH, "model_index.json")
    if not os.path.exists(config_path):
        download_model()
    else:
        print(f"✅ Model cached at {MODEL_PATH}")
    
    print("⏳ Loading pipeline...")
    
    from diffusers import LTX2Pipeline
    pipe = LTX2Pipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True
    )
    print("✅ Loaded LTX2Pipeline")
    
    pipe.enable_model_cpu_offload()
    print("✅ Ready with CPU offload!")
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
            num_frames = min(calculated_frames, 257)

        pipeline = load_model()
        
        print(f"🎬 {prompt[:50]}... ({width}x{height}, {num_frames}f)")
        
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
        
        print(f"✅ {gen_time:.1f}s")
        
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
            "frames": num_frames
        }

    except Exception as e:
        import traceback
        print(f"❌ {e}\n{traceback.format_exc()}")
        return {"error": str(e)}
    finally:
        for f in temp_files:
            if f and os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
