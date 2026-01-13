"""
LTX-2 RunPod Serverless Worker
Version: 7.0.0-Q8-RTX5090
Optimized for RTX 5090 with Q8 Kernels
Expected: ~10-15 seconds per video
"""

import runpod
import torch
import base64
import tempfile
import os
import time
import gc
import requests
import subprocess
from pathlib import Path

VERSION = "7.0.0-Q8-RTX5090"
VOLUME_PATH = "/runpod-volume"
MODEL_PATH = f"{VOLUME_PATH}/models/LTX-2-Q8"

# Global pipeline
pipe = None


def get_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        }
    return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0}


def clear_memory():
    """Clear CUDA memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def find_best_path():
    """Find the best path for model storage"""
    paths = [VOLUME_PATH, "/workspace", "/root", "/tmp"]
    for path in paths:
        if os.path.exists(path):
            try:
                stat = os.statvfs(path)
                free = stat.f_bavail * stat.f_frsize
                free_gb = free // (1024**3)
                print(f"💾 Checking {path}: {free_gb}GB free")
                if free_gb > 50:  # Need at least 50GB
                    print(f"✅ Using volume: {path}")
                    return path
            except Exception as e:
                print(f"⚠️ Error checking {path}: {e}")
    print("⚠️ Using /tmp as fallback")
    return "/tmp"


def download_q8_weights(model_path):
    """Download Q8 quantized weights"""
    from huggingface_hub import snapshot_download, hf_hub_download
    
    os.makedirs(model_path, exist_ok=True)
    
    # Download Q8 transformer weights
    print("⬇️ Downloading Q8 transformer weights...")
    hf_hub_download(
        repo_id="konakona/ltxvideo_q8",
        filename="ltxvideo_q8.safetensors",
        local_dir=model_path,
    )
    
    # Download VAE and text encoder from original repo
    print("⬇️ Downloading VAE and text encoder...")
    snapshot_download(
        repo_id="Lightricks/LTX-Video",
        local_dir=f"{model_path}/base",
        allow_patterns=["vae/*", "text_encoder/*", "tokenizer/*", "scheduler/*", "model_index.json"],
        ignore_patterns=["*.md", "*.txt", ".gitattributes", "transformer/*"]
    )
    
    print("✅ Q8 weights downloaded!")


def download_model_minimal(model_path):
    """Download full model with minimal files"""
    from huggingface_hub import snapshot_download
    
    print(f"⬇️ Downloading LTX-2 to {model_path}...")
    
    os.makedirs(model_path, exist_ok=True)
    
    snapshot_download(
        repo_id="Lightricks/LTX-2",
        local_dir=model_path,
        ignore_patterns=[
            "*.md", 
            "*.txt", 
            ".gitattributes",
            "examples/*",
            "docs/*",
            "*.png",
            "*.jpg",
            "*.gif"
        ]
    )
    
    print("✅ Download complete!")


def is_model_valid(model_path):
    """Check if model files are valid"""
    required_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
    ]
    
    required_dirs = ["text_encoder", "transformer", "vae"]
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            print(f"❌ Missing: {file}")
            return False
    
    for dir_name in required_dirs:
        dir_path = os.path.join(model_path, dir_name)
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_name}")
            return False
        
        safetensors = [f for f in os.listdir(dir_path) if f.endswith('.safetensors')]
        if not safetensors:
            print(f"❌ No safetensors in: {dir_name}")
            return False
    
    print("✅ Model files valid")
    return True


def load_model():
    """Load LTX-2 pipeline optimized for RTX 5090"""
    global pipe
    if pipe is not None:
        return pipe
    
    from diffusers import LTX2Pipeline, AutoModel
    
    base_path = find_best_path()
    model_path = f"{base_path}/models/LTX-2"
    
    # Check/download model
    if os.path.exists(model_path) and is_model_valid(model_path):
        print(f"✅ Model cached at {model_path}")
    else:
        if os.path.exists(model_path):
            print("🗑️ Removing incomplete model...")
            import shutil
            shutil.rmtree(model_path, ignore_errors=True)
        download_model_minimal(model_path)
    
    print(f"Memory before load: {get_memory_info()}")
    
    # Check GPU architecture
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
    is_ada_or_newer = any(x in gpu_name.lower() for x in ['4090', '4080', '4070', '5090', '5080', '5070', 'ada', 'blackwell'])
    
    print(f"🖥️ GPU: {gpu_name}")
    print(f"🔧 ADA/Blackwell architecture: {is_ada_or_newer}")
    
    if is_ada_or_newer:
        print("⏳ Loading pipeline with FP8 optimization for RTX 5090...")
        
        # Load transformer with FP8 for RTX 5090
        transformer = AutoModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )
        
        # Enable FP8 - works great on Blackwell/Ada
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16
        )
        print("✅ FP8 layerwise casting enabled!")
        
        pipe = LTX2Pipeline.from_pretrained(
            model_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
    else:
        print("⏳ Loading pipeline (standard mode)...")
        pipe = LTX2Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
    
    # Move to GPU
    pipe = pipe.to("cuda")
    
    # Enable memory optimizations for 32GB VRAM
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print("✅ VAE tiling/slicing enabled for memory efficiency")
    
    print(f"✅ Loaded to GPU!")
    print(f"Memory after load: {get_memory_info()}")
    
    return pipe


def get_audio_duration(audio_path):
    """Get audio duration using ffprobe"""
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
           "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def download_file(url, output_path):
    """Download file from URL"""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def save_final_output(video_frames, input_audio_path, generated_audio_waveform, output_path, fps=24):
    """Save video with optional audio"""
    from diffusers.utils import export_to_video
    import numpy as np
    import soundfile as sf
    
    temp_video = output_path.replace(".mp4", "_visual.mp4")
    temp_audio = None
    
    # Convert frames
    if torch.is_tensor(video_frames):
        video_frames = video_frames.cpu().float().numpy()
    
    if isinstance(video_frames, np.ndarray) and video_frames.ndim == 4:
        if video_frames.shape[1] in [1, 3, 4]:
            video_frames = np.transpose(video_frames, (0, 2, 3, 1))
        if video_frames.dtype in [np.float32, np.float64]:
            video_frames = (video_frames * 255).clip(0, 255).astype(np.uint8)
    
    export_to_video(video_frames, temp_video, fps=fps)
    
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_video]
    
    if input_audio_path and os.path.exists(input_audio_path):
        ffmpeg_cmd.extend(["-i", input_audio_path, "-c:v", "copy", "-c:a", "aac", 
                          "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
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


def handler(event):
    """Main handler for RunPod serverless"""
    temp_files = []
    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt")
        image_url = job_input.get("image_url")
        audio_url = job_input.get("audio_url")
        
        if not prompt:
            return {"error": "Missing prompt"}
        
        # Parameters optimized for RTX 5090
        width = job_input.get("width", 704)
        height = job_input.get("height", 480)
        fps = job_input.get("fps", 24)
        num_frames = job_input.get("num_frames", 97)
        num_inference_steps = job_input.get("num_inference_steps", 15)
        guidance_scale = job_input.get("guidance_scale", 3.5)
        
        input_audio_path = None
        
        # Handle audio
        if audio_url:
            input_audio_path = tempfile.mktemp(suffix=".mp3")
            temp_files.append(input_audio_path)
            download_file(audio_url, input_audio_path)
            
            duration = get_audio_duration(input_audio_path)
            calculated_frames = int(duration * fps) + 8
            calculated_frames = calculated_frames - (calculated_frames % 8) + 1
            num_frames = min(calculated_frames, 161)

        pipeline = load_model()
        
        mem_info = get_memory_info()
        print(f"🎬 {prompt[:50]}... ({width}x{height}, {num_frames}f, {num_inference_steps} steps)")
        print(f"Memory: {mem_info}")
        
        clear_memory()
        
        start = time.time()
        with torch.inference_mode():
            output = pipeline(
                prompt=prompt,
                negative_prompt=job_input.get("negative_prompt", "low quality, blurry, distorted"),
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(
                    job_input.get("seed", int(time.time()) % 1000000)
                ),
            )
        
        gen_time = time.time() - start
        print(f"✅ Generation: {gen_time:.1f}s")
        
        # Save output
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = f.name
            temp_files.append(out_path)
        
        generated_audio = output.audio[0] if hasattr(output, 'audio') and output.audio is not None else None
        save_final_output(output.frames[0], input_audio_path, generated_audio, out_path, fps)
        
        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        
        clear_memory()
        
        return {
            "video_base64": video_b64,
            "generation_time": round(gen_time, 2),
            "frames": num_frames,
            "resolution": f"{width}x{height}",
            "steps": num_inference_steps,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown",
            "memory": get_memory_info(),
            "version": VERSION
        }

    except Exception as e:
        import traceback
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        clear_memory()
        return {"error": str(e), "traceback": error_msg}
    finally:
        for f in temp_files:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass


def warmup():
    """Prepare for inference"""
    print("🔥 Warming up...")
    clear_memory()
    print(f"✅ Ready! Memory: {get_memory_info()}")


# === STARTUP ===
if __name__ == "__main__":
    print("=" * 60)
    print(f"🚀 LTX-2 Worker {VERSION}")
    print("=" * 60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🖥️ GPU: {gpu_name}")
        print(f"📊 VRAM: {get_memory_info()['total_gb']}GB")
        print(f"🔧 PyTorch: {torch.__version__}")
        print(f"🔧 CUDA: {torch.version.cuda}")
    else:
        print("⚠️ No CUDA available!")
    
    # Load model
    load_model()
    
    # Warmup
    warmup()
    
    print("=" * 60)
    print("✅ Worker ready!")
    print("=" * 60)
    
    runpod.serverless.start({"handler": handler})
