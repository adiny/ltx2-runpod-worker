import os
import torch
import runpod
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any

# יבוא ספריות המודל (בהנחה שאתה משתמש ב-Diffusers או בפורק הרשמי של LTX)
# הערה: אם הספרייה הרשמית שונה, יש להתאים את ה-Import
from diffusers import LTXPipeline, LTXVideoTransformer3DModel
from transformers import AutoTokenizer, T5EncoderModel

# הגדרות נתיבים ל-Volume של Runpod
# זה קריטי כדי לא להוריד 50GB בכל הפעלה מחדש
VOLUME_PATH = "/runpod-volume/LTX-2"
MODEL_ID = "Lightricks/LTX-2"

# משתנה גלובלי להחזקת המודל בזיכרון
pipe = None

def download_model_if_needed():
    """
    בודק אם המשקלים קיימים ב-Volume. אם לא, מוריד אותם בצורה חכמה.
    משתמש ב-snapshot_download כדי לתמוך בחידוש הורדה.
    """
    if not os.path.exists(VOLUME_PATH):
        print(f"Model not found in {VOLUME_PATH}. Downloading from HuggingFace...")
        from huggingface_hub import snapshot_download
        
        os.makedirs(VOLUME_PATH, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=VOLUME_PATH,
            # מסננים קבצים לא רלוונטיים כדי לחסוך מקום
            ignore_patterns=["*.msgpack", "*.bin", "*.h5"], 
            local_dir_use_symlinks=False
        )
        print("Download complete.")
    else:
        print(f"Model found in {VOLUME_PATH}. Skipping download.")

def initialize_pipeline():
    """
    טוען את המודל ל-GPU. פונקציה זו רצה פעם אחת בעת עליית הקונטיינר.
    """
    global pipe
    
    print("Loading LTX-2 components...")
    
    # טעינת המודל מה-Volume המקומי
    # אנו משתמשים ב-float16 לחיסכון בזיכרון ומהירות (Flash Attention)
    pipe = LTXPipeline.from_pretrained(
        VOLUME_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # אופטימיזציות זיכרון קריטיות ל-Runpod
    # מפעיל VAE Slicing כדי למנוע קריסת זיכרון בעת פענוח וידאו ארוך
    pipe.enable_vae_slicing()
    
    # העברה ל-GPU
    pipe.to("cuda")
    
    # הידור אופציונלי (אם נתמך בסביבה) לשיפור ביצועים של כ-20%
    # pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
    
    print("LTX-2 Model loaded successfully!")

def save_audio_video(video_frames, audio_waveform, output_path, fps=24):
    """
    מאחד את הוידאו והאודיו לקובץ MP4 אחד.
    LTX-2 מוציא אודיו ב-24kHz סטריאו.
    """
    import imageio_ffmpeg
    
    # שמירת האודיו זמנית כ-WAV (24kHz Stereo)
    temp_audio_path = output_path.replace(".mp4", ".wav")
    # המרת ה-Tensor ל-Numpy במידת הצורך
    if torch.is_tensor(audio_waveform):
        audio_waveform = audio_waveform.cpu().float().numpy()
        
    # כתיבת קובץ האודיו (Assuming shape is [Channels, Samples] or [Samples, Channels])
    # ה-Vocoder של LTX מוציא סטריאו
    sf.write(temp_audio_path, audio_waveform.T, 24000) 
    
    # שמירת הוידאו זמנית
    temp_video_path = output_path.replace(".mp4", "_silent.mp4")
    from diffusers.utils import export_to_video
    export_to_video(video_frames, temp_video_path, fps=fps)
    
    # איחוד באמצעות FFmpeg
    # משתמשים ב-AAC לאודיו ו-H.264 לוידאו לתאימות מקסימלית
    command = [
        "ffmpeg", "-y",
        "-i", temp_video_path,
        "-i", temp_audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        "-shortest", # חותך לפי הקצר מביניהם למניעת מסך שחור/שקט
        output_path
    ]
    
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # ניקוי קבצים זמניים
    os.remove(temp_audio_path)
    os.remove(temp_video_path)

def handler(event: Dict[str, Any]):
    """
    הפונקציה הראשית שמטפלת בכל בקשה (Request) מ-Runpod.
    """
    global pipe
    
    # 1. חילוץ פרמטרים מהבקשה
    job_input = event.get("input", {})
    
    prompt = job_input.get("prompt")
    negative_prompt = job_input.get("negative_prompt", "low quality, worst quality, deformed, distorted")
    image_url = job_input.get("image_url") # אופציונלי: עבור Image-to-Video
    
    # פרמטרים מתקדמים ל-LTX
    width = job_input.get("width", 768)
    height = job_input.get("height", 512)
    num_frames = job_input.get("num_frames", 121) # ~5 שניות ב-24fps
    num_inference_steps = job_input.get("num_inference_steps", 50)
    guidance_scale = job_input.get("guidance_scale", 3.0) # st (Text Guidance)
    audio_guidance_scale = job_input.get("audio_guidance_scale", 3.0) # sm (Cross-Modal Guidance)
    seed = job_input.get("seed", None)

    if not prompt and not image_url:
        return {"error": "Must provide 'prompt' or 'image_url'"}

    # הגדרת Seed לשחזור תוצאות
    generator = torch.Generator("cuda").manual_seed(seed) if seed else None

    # 2. הרצת המודל (Inference)
    # הערה: אנו קוראים לצינור שמחזיר גם וידאו וגם אודיו
    try:
        # אם יש תמונת מקור, טוענים אותה (לוגיקה בסיסית)
        image = None
        if image_url:
            from diffusers.utils import load_image
            image = load_image(image_url)

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            # פרמטרים ספציפיים ל-LTX אודיו (בהתאם למימוש הספריה)
            audio_guidance_scale=audio_guidance_scale, 
            generator=generator,
            output_type="pt" # מחזיר Tensors גולמיים לעיבוד
        )
        
        # 3. שמירת התוצאה
        output_filename = f"/tmp/ltx_out_{event['id']}.mp4"
        
        # ההנחה: הפלט מכיל 'frames' ו-'audio'
        # ה-Audio Waveform מגיע מה-Vocoder ב-24kHz
        save_audio_video(
            output.frames[0], 
            output.audio[0], 
            output_filename
        )

        # 4. העלאה לאחסון (אופציונלי - כאן מחזירים Base64 או URL)
        # עבור Runpod Sync, לרוב מחזירים Base64 לקבצים קטנים או מעלים ל-Bucket
        # כאן נדגים החזרה של נתיב (אם משתמשים ב-Volume משותף) או קידוד
        
        # בדוגמה זו: נקרא ותחזיר כ-Base64 (פחות מומלץ לוידאו ארוך, עדיף להעלות ל-S3)
        with open(output_filename, "rb") as video_file:
            video_b64 = base64.b64encode(video_file.read()).decode('utf-8')
            
        # ניקוי
        os.remove(output_filename)

        return {
            "status": "success",
            "video_base64": video_b64,
            "metadata": {
                "duration": num_frames / 24.0,
                "audio_sample_rate": 24000,
                "channels": "stereo"
            }
        }

    except Exception as e:
        print(f"Inference error: {str(e)}")
        return {"error": str(e)}

# ----------------------------------------------------------------------------
# Runpod Entry Point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. שלב ה-Build/Init: הורדת מודל
    download_model_if_needed()
    
    # 2. שלב הטעינה: טעינה לזיכרון
    initialize_pipeline()
    
    # 3. התחלת ה-Serverless Handler
    runpod.serverless.start({"handler": handler})
