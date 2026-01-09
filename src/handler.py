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

# Global pipeline - loaded once when worker starts
pipeline = None
MODEL_ID = "Lightricks/LTX-Video"


def load_model():
    """Download and load the LTX-2 model"""
    global pipeline
    
    if pipeline is not None:
        return pipeline
    
    print("🚀 Loading LTX-2 model...")
    start_time = time.time()
    
    try:
        from diffusers import LTXPipeline
        
        pipeline = LTXPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        pipeline.to("cuda")
        
        # Enable memory optimizations
        pipeline.enable_model_cpu_offload()
        
        print(f"✅ Model loaded in {time.time() - start_time:.2f} seconds")
        return pipeline
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def generate_video(pipe, params):
    """Generate video using LTX-2"""
    
    prompt = params.get("prompt", "")
    negative_prompt = params.get("negative_prompt", "blurry, low quality, distorted, watermark")
    width = params.get("width", 704)
    height = params.get("height", 480)
    num_frames = params.get("num_frames", 121)
    num_inference_steps = params.get("steps", 40)
    guidance_scale = params.get("guidance_scale", 7.5)
    seed = params.get("seed", -1)
    
    # Set seed
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    print(f"🎬 Generating: {width}x{height}, {num_frames} frames")
    print(f"📝 Prompt: {prompt[:80]}...")
    
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
    
    return output, seed, generation_time


def frames_to_video(frames, fps, output_path):
    """Convert frames to MP4 video"""
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
        return True
    except Exception as e:
        print(f"Primary export failed: {e}, trying fallback...")
        try:
            import imageio
            imageio.mimwrite(output_path, frames, fps=fps, codec='libx264')
            return True
        except Exception as e2:
            print(f"Fallback export failed: {e2}")
            raise


def handler(event):
    """RunPod serverless handler"""
    try:
        input_data = event.get("input", {})
        
        # Validate
        if not input_data.get("prompt"):
            return {"error": "Missing required field: prompt"}
        
        # Load model
        pipe = load_model()
        
        # Generate
        output, seed, generation_time = generate_video(pipe, input_data)
        
        # Get frames
        frames = output.frames[0] if hasattr(output, 'frames') else output.images
        
        # Export to temp file
        fps = input_data.get("fps", 24)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name
        
        frames_to_video(frames, fps, output_path)
        
        # Read and encode
        with open(output_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Cleanup
        os.remove(output_path)
        torch.cuda.empty_cache()
        
        return {
            "video_base64": video_base64,
            "seed": seed,
            "generation_time_seconds": round(generation_time, 2),
            "parameters": {
                "width": input_data.get("width", 704),
                "height": input_data.get("height", 480),
                "num_frames": input_data.get("num_frames", 121),
                "fps": fps,
                "steps": input_data.get("steps", 40),
            }
        }
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "GPU out of memory. Try smaller resolution or fewer frames."}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
