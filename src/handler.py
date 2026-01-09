"""
LTX Video Generation Handler
"""

import runpod
import torch
import base64
import tempfile
import os
import time

pipeline = None
MODEL_PATH = "/runpod-volume/ltx-video"

def load_model():
    global pipeline
    if pipeline is not None:
        return pipeline
    
    print("🚀 Loading model...")
    
    from huggingface_hub import snapshot_download
    from diffusers import DiffusionPipeline
    
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model...")
        snapshot_download(
            repo_id="Lightricks/LTX-Video",
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False
        )
    
    print(f"CUDA: {torch.cuda.is_available()}")
    
    pipeline = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
    )
    pipeline.to("cuda")
    
    print("✅ Model loaded!")
    return pipeline

def handler(event):
    try:
        input_data = event.get("input", {})
        prompt = input_data.get("prompt")
        
        if not prompt:
            return {"error": "Missing prompt"}
        
        width = input_data.get("width", 512)
        height = input_data.get("height", 320)
        num_frames = input_data.get("num_frames", 41)
        steps = input_data.get("steps", 30)
        fps = input_data.get("fps", 24)
        
        print(f"🎬 Generating: {prompt[:50]}...")
        
        pipe = load_model()
        
        start = time.time()
        output = pipe(
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
