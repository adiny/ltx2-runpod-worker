import runpod

def handler(event):
    prompt = event.get("input", {}).get("prompt", "No prompt")
    return {"message": f"Received: {prompt}", "status": "ok"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
