VERSION = "1.2.0"

import os
import torch
import runpod
import base64
import tempfile
import time

# Fallback imports
try:
    from diffusers import LTXPipeline
    PipelineClass = LTXPipeline
    print("Using LTXPipeline")
except ImportError:
    from diffusers import DiffusionPipeline
    PipelineClass = DiffusionPipeline
    print("Using DiffusionPipeline (fallback)")

pipe = None
VOLUME_PATH = "/runpod-volume/LTX-Video"
MODEL_ID = "Lightricks/LTX-Video"

def load_model():
    global pipe
    if pipe is not None:
        return pipe
    
    print(f"🚀 Worker Version: {VERSION}")
    print("Loading model...")
    
    from huggingface_hub import snapshot_download
    
    if not os.path.exists(VOLUME_PATH):
        print(f"📥 Downloading model to {VOLUME_PATH}...")
        os.makedirs(VOLUME_PATH, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=VOLUME_PATH,
            local_dir_use_symlinks=False
        )
    
    pipe = PipelineClass.from_pretrained(
        VOLUME_PATH,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    
    print("✅ Model loaded!")
    return pipe

def handler(event):
    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt")
        
        if not prompt:
            return {"error": "Missing prompt"}
        
        width = job_input.get("width", 512)
        height = job_input.get("height", 320)
        num_frames = job_input.get("num_frames", 41)
        steps = job_input.get("num_inference_steps", 30)
        fps = job_input.get("fps", 24)
        
        print(f"🎬 Generating: {prompt[:50]}...")
        
        pipeline = load_model()
        
        start = time.time()
        output = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
        )
        gen_time = time.time() - start
        
        print(f"✅ Done in {gen_time:.1f}s")
        
        from diffusers.utils import export_to_video
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        
        export_to_video(output.frames[0], path, fps=fps)
        
        with open(path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        
        os.remove(path)
        torch.cuda.empty_cache()
        
        return {
            "video_base64": video_b64,
            "generation_time": round(gen_time, 2)
        }
        
    except Exception as e:
        import traceback
        print(f"❌ {e}\n{traceback.format_exc()}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
