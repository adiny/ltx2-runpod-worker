VERSION = "4.2.0-FULL-MODEL"

import os
import torch
import runpod
import base64
import tempfile
import time
import subprocess
import soundfile as sf
import requests

# Use container disk for model storage
MODEL_DIR = "/workspace/models"
MODEL_PATH = f"{MODEL_DIR}/LTX-2"
CACHE_DIR = f"{MODEL_DIR}/.cache"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TMPDIR"] = "/workspace/tmp"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

pipe = None


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
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs("/workspace/tmp", exist_ok=True)
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage("/workspace")
    print(f"💾 Disk space: {free // (1024**3)} GB free / {total // (1024**3)} GB total")
    
    # Download full model if not exists
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print("📥 Downloading LTX-2 (full model)...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Lightricks/LTX-2",
            local_dir=MODEL_PATH,
            ignore_patterns=["*.md", "*.git*"],
        )
        print("✅ Download complete!")
    else:
        print(f"✅ Using cached model from {MODEL_PATH}")
    
    print("Loading pipeline...")
    from diffusers import LTXPipeline
    pipe = LTXPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    )
    print("✅ Loaded LTXPipeline")

    pipe.enable_model_cpu_offload()
    print("✅ Model ready!")
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
            input_audio_path = tempfile.mktemp(suffix=".mp3")
            temp_files.append(input_audio_path)
            download_audio(audio_url, input_audio_path)
            
            duration = get_audio_duration(input_audio_path)
            calculated_frames = int(duration * fps) + 8
            calculated_frames = calculated_frames - (calculated_frames % 8) + 1
            
            print(f"🎙️ Audio: {duration:.2f}s -> {calculated_frames} frames")
            num_frames = min(calculated_frames, 257)

        pipeline = load_model()
        
        print(f"🎬 Generating: {prompt[:50]}...")
        print(f"Settings: {width}x{height}, {num_frames} frames")
        
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
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
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
