import os
import torch
import runpod
import subprocess
import soundfile as sf
import base64
from diffusers import LTXPipeline

# הגדרות נתיבים ל-Volume
VOLUME_PATH = "/runpod-volume/LTX-2"
MODEL_ID = "Lightricks/LTX-2"

pipe = None

def download_model_if_needed():
    if not os.path.exists(VOLUME_PATH):
        print(f"Model not found in {VOLUME_PATH}. Downloading LTX-2...")
        from huggingface_hub import snapshot_download
        os.makedirs(VOLUME_PATH, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=VOLUME_PATH,
            ignore_patterns=["*.msgpack", "*.bin", "*.h5"],
            local_dir_use_symlinks=False
        )
        print("Download complete.")
    else:
        print(f"Model found in {VOLUME_PATH}.")

def initialize_pipeline():
    global pipe
    print("Loading LTX-2 components...")
    pipe = LTXPipeline.from_pretrained(
        VOLUME_PATH,
        torch_dtype=torch.bfloat16, # LTX-2 מעדיף bfloat16
        variant="fp16",
        use_safetensors=True
    )
    pipe.enable_vae_slicing() # קריטי לחיסכון בזיכרון
    pipe.to("cuda")
    print("LTX-2 Model loaded!")

def save_audio_video(video_frames, audio_waveform, output_path, fps=24):
    import imageio_ffmpeg
    temp_audio = output_path.replace(".mp4", ".wav")
    temp_video = output_path.replace(".mp4", "_silent.mp4")
    
    # שמירת אודיו (24kHz Stereo)
    if torch.is_tensor(audio_waveform):
        audio_waveform = audio_waveform.cpu().float().numpy()
    sf.write(temp_audio, audio_waveform.T, 24000)
    
    # שמירת וידאו
    from diffusers.utils import export_to_video
    export_to_video(video_frames, temp_video, fps=fps)
    
    # איחוד עם FFmpeg
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_video, "-i", temp_audio,
        "-c:v", "copy", "-c:a", "aac", "-strict", "experimental",
        "-shortest", output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    os.remove(temp_audio)
    os.remove(temp_video)

def handler(event):
    global pipe
    job_input = event.get("input", {})
    prompt = job_input.get("prompt")
    
    if not prompt:
        return {"error": "Missing prompt"}

    try:
        width = job_input.get("width", 768)
        height = job_input.get("height", 512)
        num_frames = job_input.get("num_frames", 121)
        steps = job_input.get("num_inference_steps", 50)
        
        # הרצת המודל
        output = pipe(
            prompt=prompt,
            negative_prompt=job_input.get("negative_prompt", "low quality"),
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            output_type="pt"
        )
        
        output_path = f"/tmp/ltx_{event['id']}.mp4"
        save_audio_video(output.frames[0], output.audio[0], output_path)
        
        with open(output_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
            
        os.remove(output_path)
        return {"video_base64": video_b64}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    download_model_if_needed()
    initialize_pipeline()
    runpod.serverless.start({"handler": handler})
