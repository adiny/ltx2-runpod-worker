import os

# הפניית Cache ל-Volume
os.environ["HF_HOME"] = "/runpod-volume/.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/runpod-volume/.cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

VERSION = "3.0.1-LTX2-FP8"

import torch
import runpod
import base64
import tempfile
import time
import subprocess

pipe = None
REPO_ID = "Lightricks/LTX-2"
VOLUME_PATH = "/runpod-volume/LTX-2"

# ... שאר הקוד נשאר אותו דבר ...

def save_output(video_frames, audio_waveform, output_path, fps=24):
    """שמירת וידאו + אודיו מאוחדים"""
    import soundfile as sf
    from diffusers.utils import export_to_video
    
    temp_video = output_path.replace(".mp4", "_visual.mp4")
    temp_audio = output_path.replace(".mp4", ".wav")
    
    export_to_video(video_frames, temp_video, fps=fps)
    
    if audio_waveform is not None:
        if torch.is_tensor(audio_waveform):
            audio_waveform = audio_waveform.cpu().float().numpy()
        if audio_waveform.ndim > 1 and audio_waveform.shape[0] < audio_waveform.shape[1]:
            audio_waveform = audio_waveform.T
        sf.write(temp_audio, audio_waveform, 24000)
        
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_video, "-i", temp_audio,
            "-c:v", "copy", "-c:a", "aac", "-strict", "experimental",
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        os.remove(temp_audio)
    else:
        os.rename(temp_video, output_path)
        
    if os.path.exists(temp_video):
        os.remove(temp_video)

def load_model():
    global pipe
    if pipe is not None:
        return pipe
    
    print(f"🚀 Version: {VERSION}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    from huggingface_hub import snapshot_download
    
    if not os.path.exists(VOLUME_PATH):
        print("📥 Downloading LTX-2...")
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=VOLUME_PATH,
            allow_patterns=["*fp8*", "*.json", "*.txt", "tokenizer*", "text_encoder*"],
            local_dir_use_symlinks=False
        )
    else:
        print(f"✅ Using cached model")
    
    print("Loading pipeline...")
    try:
        from diffusers import LTXPipeline
        pipe = LTXPipeline.from_pretrained(
            VOLUME_PATH,
            torch_dtype=torch.bfloat16,
            variant="fp8",
            use_safetensors=True
        )
    except Exception as e:
        print(f"⚠️ LTXPipeline failed: {e}")
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            VOLUME_PATH,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )

    pipe.enable_model_cpu_offload()
    print("✅ Model loaded!")
    return pipe

def handler(event):
    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt")
        
        if not prompt:
            return {"error": "Missing prompt"}
        
        width = job_input.get("width", 768)
        height = job_input.get("height", 512)
        num_frames = job_input.get("num_frames", 121)
        steps = job_input.get("num_inference_steps", 30)
        fps = job_input.get("fps", 24)
        
        print(f"🎬 Generating: {prompt[:50]}...")
        
        pipeline = load_model()
        
        start = time.time()
        output = pipeline(
            prompt=prompt,
            negative_prompt=job_input.get("negative_prompt", "low quality"),
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            output_type="pt"
        )
        gen_time = time.time() - start
        
        print(f"✅ Done in {gen_time:.1f}s")
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = f.name
        
        audio = output.audio[0] if hasattr(output, 'audio') and output.audio is not None else None
        save_output(output.frames[0], audio, out_path, fps)
        
        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        
        os.remove(out_path)
        torch.cuda.empty_cache()
        
        return {
            "video_base64": video_b64,
            "generation_time": round(gen_time, 2),
            "has_audio": audio is not None
        }
        
    except Exception as e:
        import traceback
        print(f"❌ {e}\n{traceback.format_exc()}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
