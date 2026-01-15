VERSION = "5.16.0-STABLE"  # Bumped version for fix

import os
import sys
import gc
import torch
import runpod
import base64
import tempfile
import time
import subprocess
import soundfile as sf
import requests
import shutil
import fcntl  # For file locking

# Disable XET and optimize memory
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

pipe = None
HF_REPO = "Lightricks/LTX-2"

# Use persistent volume path
VOLUME_PATH = "/runpod-volume"
MODEL_PATH = os.path.join(VOLUME_PATH, "models", "LTX-2")
LOCK_FILE = os.path.join(VOLUME_PATH, "models", ".ltx2_download.lock")
DOWNLOAD_TIMEOUT = 900  # 15 minutes - max time to wait for another worker's download


def get_memory_info():
    """Get current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {"allocated_gb": round(allocated, 2), "reserved_gb": round(reserved, 2), "total_gb": round(total, 2)}
    return {}


def clear_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def find_best_path():
    """Find path with most free space - prefer volume"""
    # Always prefer runpod-volume if it exists
    if os.path.exists(VOLUME_PATH):
        try:
            total, used, free = shutil.disk_usage(VOLUME_PATH)
            free_gb = free // (1024**3)
            print(f"💾 Using volume: {VOLUME_PATH} ({free_gb}GB free)")
            return VOLUME_PATH
        except:
            pass
    
    # Fallback to other paths
    paths_to_try = ["/workspace", "/root", "/tmp"]
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


def get_missing_files(model_path):
    """
    Check which files are missing WITHOUT deleting anything.
    Returns tuple: (missing_required, missing_safetensors, has_corruption)
    """
    required_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "transformer/config.json",
        "vae/config.json",
    ]
    
    tokenizer_files = ["tokenizer/tokenizer.model", "tokenizer/tokenizer_config.json"]
    safetensor_dirs = ["text_encoder", "transformer", "vae"]
    
    missing_required = []
    missing_safetensors = []
    has_corruption = False
    
    # Check required config files
    for f in required_files:
        full_path = os.path.join(model_path, f)
        if not os.path.exists(full_path):
            missing_required.append(f)
        elif os.path.getsize(full_path) == 0:
            # Empty file = corruption
            print(f"   ⚠️ Empty/corrupted: {f}")
            has_corruption = True
    
    # Check tokenizer
    tokenizer_found = any(
        os.path.exists(os.path.join(model_path, tf)) 
        for tf in tokenizer_files
    )
    if not tokenizer_found:
        missing_required.append("tokenizer/*")
    
    # Check safetensors
    for dir_name in safetensor_dirs:
        dir_path = os.path.join(model_path, dir_name)
        if not os.path.exists(dir_path):
            missing_safetensors.append(f"{dir_name}/*.safetensors")
            continue
        
        safetensor_files = [f for f in os.listdir(dir_path) if f.endswith('.safetensors')]
        if not safetensor_files:
            missing_safetensors.append(f"{dir_name}/*.safetensors")
            continue
        
        # Check for corrupted safetensors (< 1KB = definitely bad)
        for sf_file in safetensor_files:
            sf_path = os.path.join(dir_path, sf_file)
            if os.path.getsize(sf_path) < 1024:
                print(f"   ⚠️ Corrupted safetensor: {dir_name}/{sf_file}")
                has_corruption = True
    
    return missing_required, missing_safetensors, has_corruption


def is_model_valid(model_path):
    """
    Check if model files are complete and valid.
    DOES NOT DELETE anything - just reports status.
    """
    if not os.path.exists(model_path):
        return False, "not_found"
    
    print("🔍 Validating model files...")
    missing_required, missing_safetensors, has_corruption = get_missing_files(model_path)
    
    if has_corruption:
        print("   ❌ Corruption detected")
        return False, "corrupted"
    
    if missing_required:
        print(f"   ❌ Missing required: {missing_required}")
        return False, "incomplete"
    
    if missing_safetensors:
        print(f"   ❌ Missing safetensors: {missing_safetensors}")
        return False, "incomplete"
    
    print("   ✅ All model files valid")
    return True, "valid"


def download_model(model_path, force_fresh=False):
    """
    Download model using huggingface_hub.
    Supports resuming partial downloads unless force_fresh=True.
    """
    print(f"📥 Downloading {HF_REPO}...")
    
    base_dir = os.path.dirname(model_path)
    cache_dir = os.path.join(base_dir, "hf_cache")
    
    if force_fresh:
        print("🗑️ Force fresh download - cleaning everything...")
        if os.path.exists(model_path):
            shutil.rmtree(model_path, ignore_errors=True)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
    
    # Create directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    
    from huggingface_hub import snapshot_download
    
    try:
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=model_path,
            cache_dir=cache_dir,
            ignore_patterns=["*.md", "*.git*", "*.mp4", "*fp4*", "*fp8*", "*distilled*", "*19b-dev.safetensors"],
            max_workers=2,
            resume_download=True,  # ✅ CRITICAL: Resume partial downloads!
        )
    except Exception as e:
        print(f"❌ Download failed: {e}")
        # DON'T delete on failure - let it resume next time
        raise
    
    # Clean cache after successful download (keep model files)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    
    print("✅ Download complete!")


def acquire_download_lock():
    """
    Acquire exclusive lock for downloading.
    Returns lock file handle if acquired, None if another worker is downloading.
    """
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    
    try:
        lock_fd = open(LOCK_FILE, 'w')
        # Try to acquire exclusive lock (non-blocking)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Write our PID and timestamp
        lock_fd.write(f"{os.getpid()}:{time.time()}")
        lock_fd.flush()
        print(f"🔒 Acquired download lock (PID: {os.getpid()})")
        return lock_fd
    except (IOError, OSError):
        # Another process has the lock
        return None


def release_download_lock(lock_fd):
    """Release the download lock"""
    if lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            print("🔓 Released download lock")
        except:
            pass


def wait_for_download_completion(model_path, timeout=DOWNLOAD_TIMEOUT):
    """
    Wait for another worker to complete the download.
    Returns True if model becomes valid, False if timeout.
    """
    print(f"⏳ Another worker is downloading. Waiting up to {timeout}s...")
    start_time = time.time()
    check_interval = 10  # Check every 10 seconds
    
    while time.time() - start_time < timeout:
        time.sleep(check_interval)
        
        # Check if model is now valid
        is_valid, status = is_model_valid(model_path)
        if is_valid:
            print("✅ Model ready (downloaded by another worker)")
            return True
        
        # Check if lock is still held
        try:
            with open(LOCK_FILE, 'r') as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
                # Lock was released - other worker finished (or crashed)
                print("🔓 Download lock released by other worker")
                break
        except (IOError, OSError):
            # Lock still held - download in progress
            elapsed = int(time.time() - start_time)
            print(f"   ... still waiting ({elapsed}s)")
            continue
    
    return False


def ensure_model_ready(model_path):
    """
    Smart model preparation with file locking to prevent race conditions:
    1. If valid -> use it
    2. If another worker is downloading -> wait
    3. If incomplete -> acquire lock and resume download
    4. If corrupted -> acquire lock and fresh download
    """
    is_valid, status = is_model_valid(model_path)
    
    if is_valid:
        print(f"✅ Model cached and valid at {model_path}")
        return True
    
    # Try to acquire download lock
    lock_fd = acquire_download_lock()
    
    if lock_fd is None:
        # Another worker is downloading - wait for it
        if wait_for_download_completion(model_path):
            return True
        
        # Timeout or other worker failed - try to acquire lock ourselves
        lock_fd = acquire_download_lock()
        if lock_fd is None:
            # Still can't get lock - re-validate and wait more
            is_valid, _ = is_model_valid(model_path)
            if is_valid:
                return True
            raise RuntimeError("Cannot acquire download lock and model is not ready")
    
    try:
        # We have the lock - re-validate (might have changed while waiting)
        is_valid, status = is_model_valid(model_path)
        if is_valid:
            print(f"✅ Model became valid while waiting for lock")
            return True
        
        if status == "corrupted":
            print("⚠️ Corruption detected - starting fresh download...")
            download_model(model_path, force_fresh=True)
        elif status == "incomplete":
            print("📥 Incomplete model - resuming download...")
            download_model(model_path, force_fresh=False)  # Resume!
        else:  # not_found
            print("📥 Model not found - starting download...")
            download_model(model_path, force_fresh=False)
        
        # Final validation
        is_valid, status = is_model_valid(model_path)
        if not is_valid:
            raise RuntimeError(f"Model at {model_path} still invalid after download! Status: {status}")
        
        return True
    
    finally:
        release_download_lock(lock_fd)


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
    import numpy as np
    
    temp_video = output_path.replace(".mp4", "_visual.mp4")
    temp_audio = None
    
    # Convert frames to proper format if needed
    if torch.is_tensor(video_frames):
        # Convert bfloat16 to float32 first (numpy doesn't support bfloat16)
        video_frames = video_frames.cpu().float().numpy()
    
    # If frames are in format (frames, channels, height, width), convert to (frames, height, width, channels)
    if isinstance(video_frames, np.ndarray) and video_frames.ndim == 4:
        if video_frames.shape[1] in [1, 3, 4]:  # channels first
            video_frames = np.transpose(video_frames, (0, 2, 3, 1))
        # Convert to uint8 if float
        if video_frames.dtype in [np.float32, np.float64]:
            video_frames = (video_frames * 255).clip(0, 255).astype(np.uint8)
    
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
    print(f"Memory: {get_memory_info()}")
    
    # Check accelerate
    try:
        import accelerate
        print(f"accelerate: {accelerate.__version__}")
    except ImportError:
        print("⚠️ accelerate not found!")
    
    # Find best path - prefer volume
    base_path = find_best_path()
    model_path = os.path.join(base_path, "models", "LTX-2")
    
    # Smart model preparation (validates, resumes, or downloads as needed)
    ensure_model_ready(model_path)
    
    # Clear memory before loading
    clear_memory()
    print(f"Memory before load: {get_memory_info()}")
    
    print("⏳ Loading pipeline...")
    
    from diffusers import LTX2Pipeline
    
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True
    )
    print("✅ Loaded LTX2Pipeline")
    
    # Load directly to GPU (no CPU offload for speed)
    pipe = pipe.to("cuda")
    print("✅ Loaded to GPU!")
    
    # Clear memory after loading
    clear_memory()
    print(f"Memory after load: {get_memory_info()}")
    
    return pipe


def warmup_model():
    """Clear memory and prepare for inference - no actual warmup to save VRAM"""
    global pipe
    
    print("🔥 Preparing model (no warmup to save VRAM)...")
    
    # Just clear memory aggressively
    clear_memory()
    
    # Force garbage collection
    import gc
    gc.collect()
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print(f"✅ Ready! Memory: {get_memory_info()}")


def handler(event):
    temp_files = []
    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt")
        audio_url = job_input.get("audio_url")
        
        if not prompt:
            return {"error": "Missing prompt"}
        
        width = job_input.get("width", 704)
        height = job_input.get("height", 480)
        fps = job_input.get("fps", 24)
        num_frames = job_input.get("num_frames", 97)
        input_audio_path = None
        
        if audio_url:
            input_audio_path = tempfile.mktemp(suffix=".mp3")
            temp_files.append(input_audio_path)
            download_audio(audio_url, input_audio_path)
            
            duration = get_audio_duration(input_audio_path)
            calculated_frames = int(duration * fps) + 8
            calculated_frames = calculated_frames - (calculated_frames % 8) + 1
            num_frames = min(calculated_frames, 161)

        pipeline = load_model()
        
        # Memory check before generation
        mem_info = get_memory_info()
        print(f"🎬 {prompt[:50]}... ({width}x{height}, {num_frames}f)")
        print(f"Memory: {mem_info}")
        
        # Clear before generation
        clear_memory()
        
        start = time.time()
        with torch.inference_mode():
            output = pipeline(
                prompt=prompt,
                negative_prompt=job_input.get("negative_prompt", "low quality"),
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=job_input.get("num_inference_steps", 15),
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
        
        # Clear after generation
        clear_memory()
        
        return {
            "video_base64": video_b64,
            "generation_time": round(gen_time, 2),
            "frames": num_frames,
            "memory": get_memory_info()
        }

    except Exception as e:
        import traceback
        print(f"❌ {e}\n{traceback.format_exc()}")
        clear_memory()
        return {"error": str(e)}
    finally:
        for f in temp_files:
            if f and os.path.exists(f):
                os.remove(f)


# === PRE-WARMING ON STARTUP ===
if __name__ == "__main__":
    print("=" * 50)
    print(f"🚀 LTX-2 Worker {VERSION}")
    print("=" * 50)
    
    # Load model at startup (before any jobs arrive)
    load_model()
    
    # Warm up with minimal inference
    warmup_model()
    
    print("=" * 50)
    print("✅ Worker ready to accept jobs!")
    print("=" * 50)
    
    # Start the serverless handler
    runpod.serverless.start({"handler": handler})
