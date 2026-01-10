VERSION = "4.0.0-BAKED-AUDIO-SYNC"

import os
import torch
import runpod
import base64
import tempfile
import time
import subprocess
import soundfile as sf
import requests

# Model is baked into the image at /models
os.environ["HF_HOME"] = "/models/.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/models/.cache"

pipe = None
MODEL_PATH = "/models/LTX-2"


def get_audio_duration(file_path):
    """חישוב משך האודיו בשניות"""
    f = sf.SoundFile(file_path)
    return len(f) / f.samplerate


def download_audio(url, save_path):
    """הורדת קובץ אודיו מ-URL"""
    print(f"📥 Downloading audio from: {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download audio. Status: {response.status_code}")


def save_final_output(video_frames, input_audio_path, generated_audio_waveform, output_path, fps=24):
    """
    איחוד חכם:
    1. אם יש אודיו מהמשתמש (input_audio) -> הוא המלך. משתמשים בו.
    2. אם אין, משתמשים באודיו של LTX (אם יש).
    """
    from diffusers.utils import export_to_video
    
    temp_video = output_path.replace(".mp4", "_visual.mp4")
    temp_audio = None
    
    # שמירת הוידאו
    export_to_video(video_frames, temp_video, fps=fps)
    
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_video]
    
    if input_audio_path and os.path.exists(input_audio_path):
        print("🎵 Merging USER audio (High Priority)...")
        ffmpeg_cmd.extend(["-i", input_audio_path, "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
        
    elif generated_audio_waveform is not None:
        print("🎵 Merging GENERATED audio (Low Priority)...")
        temp_audio = output_path.replace(".mp4", ".wav")
        if torch.is_tensor(generated_audio_waveform):
            generated_audio_waveform = generated_audio_waveform.cpu().float().numpy()
        if generated_audio_waveform.ndim > 1 and generated_audio_waveform.shape[0] < generated_audio_waveform.shape[1]:
            generated_audio_waveform = generated_audio_waveform.T
        sf.write(temp_audio, generated_audio_waveform, 24000)
        ffmpeg_cmd.extend(["-i", temp_audio, "-c:v", "copy", "-c:a", "aac", "-shortest"])
    else:
        print("🔇 No audio source. Creating silent video.")
        os.rename(temp_video, output_path)
        return

    ffmpeg_cmd.extend(["-strict", "experimental", output_path])
    
    # הרצת FFmpeg
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # ניקוי
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
    
    # Model is already in the image - no download needed!
    print(f"📦 Loading model from {MODEL_PATH}")
    
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        raise RuntimeError(f"Model not found at {MODEL_PATH}! Image was not built correctly.")
    
    print("Loading pipeline...")
    try:
        from diffusers import LTXPipeline
        pipe = LTXPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            variant="fp8",
            use_safetensors=True
        )
        print("✅ Loaded LTXPipeline (fp8)")
    except Exception as e:
        print(f"⚠️ LTXPipeline failed: {e}, trying without variant...")
        try:
            from diffusers import LTXPipeline
            pipe = LTXPipeline.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            )
            print("✅ Loaded LTXPipeline (standard)")
        except Exception as e2:
            print(f"⚠️ Fallback to DiffusionPipeline: {e2}")
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            )
            print("✅ Loaded DiffusionPipeline (fallback)")

    pipe.enable_model_cpu_offload()
    print("✅ Model loaded and ready!")
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
        
        # חישוב פריימים חכם
        num_frames = job_input.get("num_frames", 121)
        input_audio_path = None
        
        if audio_url:
            # אם יש URL, מורידים ומחשבים אורך
            input_audio_path = tempfile.mktemp(suffix=".mp3")
            temp_files.append(input_audio_path)
            download_audio(audio_url, input_audio_path)
            
            duration = get_audio_duration(input_audio_path)
            calculated_frames = int(duration * fps) + 8
            
            # LTX-2 עובד בקפיצות מסוימות (מודולו 8)
            calculated_frames = calculated_frames - (calculated_frames % 8) + 1
            
            print(f"🎙️ Audio input detected: {duration:.2f}s -> Setting frames to {calculated_frames}")
            num_frames = min(calculated_frames, 257)

        pipeline = load_model()
        
        print(f"🎬 Generating: {prompt[:50]}...")
        print(f"Settings: {width}x{height}, {num_frames} frames, {fps} fps")
        
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
        
        print(f"✅ Generation done in {gen_time:.1f}s")
        
        # שמירה ואיחוד
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
        # ניקוי קבצים
        for f in temp_files:
            if f and os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
