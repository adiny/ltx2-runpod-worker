VERSION = "5.5.0-CUDA-DIRECT"

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

# Disable XET
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

pipe = None
HF_REPO = "Lightricks/LTX-2"


def find_best_path():
    """Find path with most free space"""
    paths_to_try = ["/runpod-volume", "/workspace", "/root", "/tmp"]
    
    best_path = "/root"
    best_free = 0
    
    print("💾 Checking disk space:")
    for path in paths_to_try:
        try:
            os.makedirs(path, exist_ok=True)
            total, used, free = shutil.disk_usage(path)
            free_gb = free // (1024**3)
            print(f"   {path}: {free_gb}GB free")
            if free > best_free:
                best_free = free
                best_path = path
        except:
            pass
    
    return best_path


def download_model(model_path):
    """Download model using huggingface_hub"""
    print(f"📥 Downloading {HF_REPO}...")
    
    cache_dir = os.path.join(os.path.dirname(model_path), "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    
    from huggingface_hub import snapshot_download
    
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=model_path,
        cache_dir=cache_dir,
        ignore_patterns=["*.md", "*.git*", "*.mp4", "*fp4*", "*fp8*", "*distilled*", "*19b-dev.safetensors"],
        max_workers=2,
    )
    
    # Clean cache
    shutil.rmtree(cache_dir, ignore_errors=True)
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
    print(f"PyTorch: {torch.__version__}")
    
    # Find best path
    base_path = find_best_path()
    model_path = os.path.join(base_path, "LTX-2")
    
    # Download model if needed
    config_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(config_path):
        download_model(model_path)
    else:
        print(f"✅ Model cached at {model_path}")
    
    print("⏳ Loading pipeline...")
    
    from diffusers import LTX2Pipeline
    
    # Load with lower precision to save memory
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True
    )
    print("✅ Loaded LTX2Pipeline")
    
    # Move to GPU directly
    pipe.to("cuda")
    print("✅ Ready on GPU!")
    
    return pipe


def handler(event):
    temp_files = []
    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt")
        audio_url = job_input.get("audio_url")
        
        if not prompt:
            return {"error": "Missing prompt"}
        
        width = job_input.get("width", 704)  # Slightly smaller to save memory
        height = job_input.get("height", 480)
        fps = job_input.get("fps", 24)
        num_frames = job_input.get("num_frames", 97)  # Fewer frames to save memory
        input_audio_path = None
        
        if audio_url:
            input_audio_path = tempfile.mktemp(suffix=".mp3")
            temp_files.append(input_audio_path)
            download_audio(audio_url, input_audio_path)
            
            duration = get_audio_duration(input_audio_path)
            calculated_frames = int(duration * fps) + 8
            calculated_frames = calculated_frames - (calculated_frames % 8) + 1
            num_frames = min(calculated_frames, 161)  # Cap at ~6 seconds

        pipeline = load_model()
        
        print(f"🎬 {prompt[:50]}... ({width}x{height}, {num_frames}f)")
        
        # Clear memory before generation
        torch.cuda.empty_cache()
        
        start = time.time()
        with torch.inference_mode():
            output = pipeline(
                prompt=prompt,
                negative_prompt=job_input.get("negative_prompt", "low quality"),
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=job_input.get("num_inference_steps", 25),
                output_type="pt"
            )
        gen_time = time.time() - start
        
        print(f"✅ {gen_time:.1f}s")
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = f.name
            temp_files.append(out_path)
            
        generated_audio = output.audio[0] if hasattr(output, 'audio') and output.audio is not None else None
        save_final_output(output.frames[0], input_audio_path, generated_audio, out_path, fps)
        
        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        
        # Clear memory after generation
        torch.cuda.empty_cache()
        
        return {
            "video_base64": video_b64,
            "generation_time": round(gen_time, 2),
            "frames": num_frames
        }

    except Exception as e:
        import traceback
        print(f"❌ {e}\n{traceback.format_exc()}")
        torch.cuda.empty_cache()
        return {"error": str(e)}
    finally:
        for f in temp_files:
            if f and os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
