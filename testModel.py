import requests
import argparse
import json
import sys
import time
import curses
import textwrap

# --- Configuration ---
# Address of your vLLM OpenAI API server
VLLM_API_URL = "http://localhost:80/v1/chat/completions"
# Endpoint to fetch the model list
VLLM_MODELS_URL = "http://localhost:80/v1/models"
HEADERS = {"Content-Type": "application/json"}


def get_vllm_completion(prompt: str, model_name: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """Send a completion request to vLLM and return the assistant's reply text."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    try:
        response = requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"[Error] Request failed: {e}"

def stream_vllm(prompt: str, model_name: str, temperature: float = 0.7):
    """Yield response chunks from vLLM in streaming mode."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": temperature,
        "stream": True
    }
    try:
        with requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload), stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_lines():
                if chunk:
                    decoded = chunk.decode("utf-8")
                    if decoded.startswith("data: "):
                        json_part = decoded[len("data: "):]
                        if json_part.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(json_part)
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue
    except requests.exceptions.RequestException as e:
        yield f"[Error] Request failed: {e}"


def chat_cli(model_name: str):
    """Interactive curses-based chat UI with scrollable history."""
    import curses
    import textwrap

    def _run(stdscr):
        curses.curs_set(1)
        stdscr.keypad(True)

        max_y, max_x = stdscr.getmaxyx()
        chat_height = max_y - 3  # space for input box
        chat_pad = curses.newpad(1000, max_x)
        input_win = curses.newwin(3, max_x, chat_height, 0)
        input_win.keypad(True)

        history = []
        scroll = 0
        current_input = ""

        def redraw():
            chat_pad.clear()
            y = 0
            for msg in history:
                for wrapped in textwrap.wrap(msg, max_x - 2):
                    if y < 999:
                        chat_pad.addstr(y, 0, wrapped[:max_x - 1])
                        y += 1
            chat_pad.refresh(scroll, 0, 0, 0, chat_height - 1, max_x - 1)

        redraw()
        while True:
            input_win.clear()
            input_win.border()
            input_win.addstr(1, 2, current_input[:max_x - 4])
            input_win.refresh()

            ch = stdscr.getch()
            if ch in (10, 13):  # Enter
                prompt = current_input.strip()
                if prompt.lower() in ("exit", "quit"):
                    break
                if prompt:
                    history.append(f'You: {prompt}')
                    redraw()
                    history.append('Model: ')
                    answer_idx = len(history) - 1
                    redraw()
                    start_time = time.time()
                    tokens_seen = 0
                    for token in stream_vllm(prompt, model_name):
                        tokens_seen += len(token)
                        history[answer_idx] += token
                        scroll = max(0, len(history) - chat_height)
                        elapsed = max(time.time() - start_time, 1e-3)
                        tps = tokens_seen / elapsed
                        input_win.addstr(0, 2, f"TPS: {tps:.1f}".ljust(max_x - 4))
                        input_win.refresh()
                        redraw()
                current_input = ""
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                current_input = current_input[:-1]
            elif ch == curses.KEY_PPAGE:
                scroll = max(0, scroll - chat_height)
                redraw()
            elif ch == curses.KEY_NPAGE:
                scroll = min(max(0, len(history) - chat_height), scroll + chat_height)
                redraw()
            elif ch == 27:  # ESC
                break
            elif 32 <= ch <= 126:
                current_input += chr(ch)

    curses.wrapper(_run)

def get_first_available_model():
    """Automatically fetch the first available model name from the vLLM server."""
    try:
        response = requests.get(VLLM_MODELS_URL)
        response.raise_for_status()  # Raise an exception if the request fails
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
    parser.add_argument("prompt", nargs="?", type=str, help="The question you want to ask the large language model.")
    parser.add_argument("--stream", action="store_true", help="Receive the response in streaming mode.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start an interactive chat session (PageUp/PageDown to scroll history).")
    
    args = parser.parse_args()

    model_name = get_first_available_model()
    
    if not model_name:
        sys.exit(1) # Exit the script if no model is fetched

        # Interactive chat
    if args.interactive or args.prompt is None:
        chat_cli(model_name)
        return

    print(f"💬 Sending question: \"{args.prompt}\"")

    if args.stream:
        query_vllm_stream(args.prompt, model_name)
    else:
        query_vllm_once(args.prompt, model_name)

if __name__ == "__main__":
    main()