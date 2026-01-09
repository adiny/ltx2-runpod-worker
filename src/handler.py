"""
LTX-2 RunPod Serverless Handler
Video generation using Lightricks LTX-2 model
"""

import runpod
import torch
import os
import base64
import tempfile
import time

# Global pipeline
pipeline = None
MODEL_ID = "Lightricks/LTX-Video"


def load_model():
    """Download and load the LTX model"""
    global pipeline
    
    if pipeline is not None:
        return pipeline
    
    print("🚀 Loading LTX model...")
    start_time = time.time()
    
    try:
        from diffusers import LTXPipeline
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        pipeline = LTXPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
        )
        pipeline.to("cuda")
        
        print(f"✅ Model loaded in {time.time() - start_time:.2f} seconds")
        return pipeline
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def handler(event):
    """RunPod serverless handler"""
    try:
        input_data = event.get("input", {})
        
        # Validate
        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Missing required field: prompt"}
        
        # Parameters
        negative_prompt = input_data.get("negative_prompt", "blurry, low quality, distorted")
        width = input_data.get("width", 512)
        height = input_data.get("height", 320)
        num_frames = input_data.get("num_frames", 41)
        num_inference_steps = input_data.get("steps", 30)
        guidance_scale = input_data.get("guidance_scale", 7.5)
        fps = input_data.get("fps", 24)
        seed = input_data.get("seed", -1)
        
        # Set seed
        if seed == -1:
            seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        print(f"🎬 Generating: {width}x{height}, {num_frames} frames")
        print(f"📝 Prompt: {prompt[:80]}...")
        
        # Load model
        pipe = load_model()
        
        # Generate
        start_time = time.time()
        
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        generation_time = time.time() - start_time
        print(f"✅ Generated in {generation_time:.2f} seconds")
        
        # Get frames
        if hasattr(output, 'frames'):
            frames = output.frames[0]
        else:
            frames = output.images
        
        # Export video
        from diffusers.utils import export_to_video
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name
        
        export_to_video(frames, output_path, fps=fps)
        
        # Read and encode
        with open(output_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Cleanup
        os.remove(output_path)
        torch.cuda.empty_cache()
        
        return {
            "video_base64": video_base64,
            "seed": seed,
            "generation_time_seconds": round(generation_time, 2)
        }
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ Error: {error_msg}")
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
