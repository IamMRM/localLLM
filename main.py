import uvicorn
import subprocess
import time
import requests

def get_ngrok_url():
    time.sleep(3)  # Give ngrok a moment to start
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        public_url = response.json()["tunnels"][0]["public_url"]
        return public_url
    except:
        return None

if __name__ == "__main__":
    # Start ngrok in a separate process
    ngrok_process = subprocess.Popen(["ngrok", "http", "8000"])

    # Get the ngrok URL
    ngrok_url = get_ngrok_url()
    if ngrok_url:
        print(f"Ngrok tunnel established: {ngrok_url}")
        print(f"Gradio interface available at: {ngrok_url}/gradio")
    else:
        print("Failed to get ngrok URL. Make sure ngrok is installed and configured.")

    # Start the FastAPI app
    uvicorn.run(
        "start:app",
        reload=True,
        host="0.0.0.0",
        port=8000
    )

    # When the script is interrupted, close the ngrok process
    ngrok_process.terminate()