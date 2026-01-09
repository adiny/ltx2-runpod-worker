"""
LTX Video RunPod Serverless Handler
"""

import runpod
import torch
import os
import base64
import tempfile
import time

# Global pipeline
pipeline = None


def load_model():
    """Load the LTX Video model"""
    global pipeline
    
    if pipeline is not None:
        return pipeline
    
    print("🚀 Loading model...")
    start_time = time.time()
    
    try:
        from diffusers import DiffusionPipeline
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Load a simpler text-to-video model that works
        pipeline = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipeline.to("cuda")
        
        # Enable memory optimization
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_slicing()
        
        print(f"✅ Model loaded in {time.time() - start_time:.2f}s")
        return pipeline
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


def handler(event):
    """RunPod handler"""
    try:
        input_data = event.get("input", {})
        
        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Missing prompt"}
        
        # Parameters
        negative_prompt = input_data.get("negative_prompt", "blurry, low quality")
        num_frames = min(input_data.get("num_frames", 16), 24)  # Max 24 frames
        num_inference_steps = input_data.get("steps", 25)
        guidance_scale = input_data.get("guidance_scale", 7.5)
        fps = input_data.get("fps", 8)
        seed = input_data.get("seed", -1)
        
        if seed == -1:
            seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        print(f"🎬 Generating {num_frames} frames...")
        print(f"📝 Prompt: {prompt[:60]}...")
        
        pipe = load_model()
        
        start_time = time.time()
        
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        generation_time = time.time() - start_time
        print(f"✅ Done in {generation_time:.2f}s")
        
        # Get frames
        frames = output.frames[0]
        
        # Export
        from diffusers.utils import export_to_video
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name
        
        export_to_video(frames, output_path, fps=fps)
        
        with open(output_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        os.remove(output_path)
        torch.cuda.empty_cache()
        
        return {
            "video_base64": video_base64,
            "seed": seed,
            "generation_time_seconds": round(generation_time, 2)
        }
        
    except Exception as e:
        import traceback
        print(f"❌ {e}\n{traceback.format_exc()}")
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
