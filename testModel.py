import requests
import argparse
import json
import sys

# --- Configuration ---
# Address of your vLLM OpenAI API server
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
# Endpoint to fetch the model list
VLLM_MODELS_URL = "http://localhost:8000/v1/models"
HEADERS = {"Content-Type": "application/json"}

def get_first_available_model():
    """Automatically fetch the first available model name from the vLLM server."""
    try:
        response = requests.get(VLLM_MODELS_URL)
        response.raise_for_status()  # 如果請求失敗則拋出異常
        models = response.json()
        if "data" in models and len(models["data"]) > 0:
            model_name = models["data"][0]["id"]
            print(f"✅ Detected model automatically: {model_name}")
            return model_name
        else:
            print("❌ Error: Unable to retrieve model list from server. Please check that the server has loaded models correctly.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Unable to connect to vLLM server at {VLLM_MODELS_URL}.")
        print("Please verify:")
        print("1. Has the vLLM server started successfully?")
        print("2. Is the server address and port correct?")
        print(f"Detailed error: {e}")
        return None

def query_vllm_once(prompt: str, model_name: str):
    """Send a request and retrieve the complete response in a single call."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": False # Set to False
    }

    try:
        response = requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        
        data = response.json()
        content = data['choices'][0]['message']['content']

        print("\n--- vLLM Response (single-shot) ---")
        print(content)
        print("--------------------------\n")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def query_vllm_stream(prompt: str, model_name: str):
    """Send a request and process the response in streaming mode."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": True # Set to True
    }

    try:
        print("\n--- vLLM Response (stream) ---")
        with requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload), stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    # vLLM streaming output starts with "data: "
                    if decoded_chunk.startswith('data: '):
                        json_str = decoded_chunk[len('data: '):]
                        if json_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(json_str)
                            delta = data['choices'][0]['delta']
                            if 'content' in delta:
                                content = delta['content']
                                # Use end='' and flush=True for real-time printing
                                print(content, end='', flush=True)
                        except json.JSONDecodeError:
                            print(f"\nUnparsable JSON chunk: {json_str}")
        print("\n--------------------------\n")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="A Python script to test the vLLM OpenAI API.")
    parser.add_argument("prompt", type=str, help="The question you want to ask the large language model.")
    parser.add_argument("--stream", action="store_true", help="Receive the response in streaming mode.")
    
    args = parser.parse_args()

    model_name = get_first_available_model()
    
    if not model_name:
        sys.exit(1) # Exit the script if no model is fetched

    print(f"💬 Sending question: \"{args.prompt}\"")

    if args.stream:
        query_vllm_stream(args.prompt, model_name)
    else:
        query_vllm_once(args.prompt, model_name)

if __name__ == "__main__":
    main()