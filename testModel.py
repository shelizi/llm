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

# Default system prompt
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def get_vllm_completion(prompt: str, model_name: str, max_tokens: int = 1024, temperature: float = 0.7, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
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

def stream_vllm(prompt: str, model_name: str, temperature: float = 0.7, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
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


def chat_cli(model_name: str, max_context_tokens: int = 2048, default_system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """Interactive curses-based chat UI with scrollable history.

    Adds real-time TPS (tokens per second) statistics and displays the
    size of the context sent to the model. The context consists of the
    full conversation history truncated from the oldest message when the
    approximate token count would exceed ``max_context_tokens``.
    """
    import curses
    import textwrap

    # Current system prompt (can be updated via /system command)
    system_prompt = default_system_prompt
    # Runtime-tunable parameters
    max_tokens = 1024          # Default number of tokens the model may generate
    temperature = 0.7          # Sampling temperature (0 = deterministic)

    # --- Helper utilities (simple, fast approximations) ---
    def _estimate_tokens(text: str) -> int:
        """Rudimentary token estimation (whitespace split)."""
        return len(text.split())

    def _build_messages(conv, user_prompt: str):
        """Build message list with history and respect token budget."""
        system_msg = {"role": "system", "content": system_prompt}
        messages = [system_msg]
        # Remaining budget after adding system and upcoming user prompt
        remaining = max_context_tokens - _estimate_tokens(system_msg["content"]) - _estimate_tokens(user_prompt)
        for role, content in reversed(conv):  # newest → oldest
            t = _estimate_tokens(content)
            if remaining - t < 0:
                break
            messages.insert(1, {"role": role, "content": content})  # after system
            remaining -= t
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _stream_vllm_messages(messages):
        """Yield streamed tokens from vLLM given a full message list."""
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        try:
            with requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload), stream=True) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_lines():
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

    def _run(stdscr):
        nonlocal system_prompt, max_context_tokens, max_tokens, temperature
        curses.curs_set(1)
        curses.noecho()  # Disable automatic echoing of keys
        stdscr.keypad(True)

        # Initialize color pairs for simple Markdown highlighting
        curses.start_color()
        curses.use_default_colors()
        # Pair numbers: 1=code, 2=heading, 3=list, 4=blockquote
        curses.init_pair(1, curses.COLOR_CYAN, -1)      # Code blocks / inline code
        curses.init_pair(2, curses.COLOR_YELLOW, -1)    # Headings
        curses.init_pair(3, curses.COLOR_GREEN, -1)     # List items
        curses.init_pair(4, curses.COLOR_MAGENTA, -1)   # Blockquotes

        max_y, max_x = stdscr.getmaxyx()
        chat_height = max_y - 3  # space for input box
        chat_pad = curses.newpad(1000, max_x)
        input_win = curses.newwin(3, max_x, chat_height, 0)
        input_win.keypad(True)

        history = []          # For UI rendering
        conversation = []     # For building context (list of (role, content))
        scroll = 0              # Topmost line currently shown
        auto_scroll = True      # Whether we should keep view pinned to bottom
        total_lines = 0         # Total rendered lines in chat_pad (updated in redraw)
        current_input = ""
        last_redraw = 0  # Track last UI refresh time

        def redraw():
            """Redraw chat history pad and flush in a single screen update."""
            nonlocal total_lines, scroll
            chat_pad.erase()
            y = 0
            in_code_block = False
            for msg in history:
                # Preserve explicit newlines by splitting first
                for para in msg.split("\n"):
                    # Detect code block delimiters ```
                    if para.strip().startswith("```"):
                        in_code_block = not in_code_block
                        # Skip the delimiter line itself
                        if y < 999:
                            y += 1
                        continue

                    # Blank line handling
                    if para == "":
                        if y < 999:
                            y += 1
                        continue

                    # Choose attribute based on markdown context
                    attr = curses.A_NORMAL
                    if in_code_block:
                        attr = curses.color_pair(1)
                    elif para.startswith("#"):
                        attr = curses.color_pair(2) | curses.A_BOLD
                    elif para.lstrip().startswith(('- ','* ','+ ')):
                        attr = curses.color_pair(3)
                    elif para.lstrip().startswith('>'):
                        attr = curses.color_pair(4)
                    elif '`' in para:
                        attr = curses.color_pair(1)

                    for wrapped in textwrap.wrap(para, max_x - 2):
                        if y < 999:
                            chat_pad.addstr(y, 0, wrapped[:max_x - 1], attr)
                            y += 1
            total_lines = y
            # Adjust scroll if auto-scroll is enabled or scroll beyond limits
            scroll = max(0, min(scroll, max(0, total_lines - chat_height)))
            chat_pad.noutrefresh(scroll, 0, 0, 0, chat_height - 1, max_x - 1)
            curses.doupdate()

        redraw()
        while True:
            input_win.erase()
            input_win.border()
            # Show only the tail part if input is wider than window
            trimmed = current_input[-(max_x - 4):]
            input_win.addstr(1, 2, trimmed)
            input_win.noutrefresh()
            curses.doupdate()

            ch = stdscr.getch()
            if ch in (10, 13):  # Enter
                prompt = current_input.strip()
                if prompt.lower() in ("exit", "quit"):
                    break
                if prompt:
                    # Handle runtime configuration commands
                    if prompt.lower().startswith("/system"):
                        new_prompt = prompt[len("/system"):].strip()
                        if new_prompt:
                            system_prompt = new_prompt
                            history.append(f"[System prompt updated] {system_prompt}")
                        else:
                            system_prompt = default_system_prompt
                            history.append(f"[System prompt reset to default] {system_prompt}")
                        redraw()
                        current_input = ""
                        continue
                    elif prompt.lower().startswith("/max_context"):
                        value = prompt[len("/max_context"):].strip()
                        if value.isdigit():
                            max_context_tokens = int(value)
                            history.append(f"[max_context_tokens updated] {max_context_tokens}")
                        else:
                            history.append("[Error] Invalid value for max_context_tokens (expect integer)")
                        redraw()
                        current_input = ""
                        continue
                    elif prompt.lower().startswith("/max_tokens"):
                        value = prompt[len("/max_tokens"):].strip()
                        if value.isdigit():
                            max_tokens = int(value)
                            history.append(f"[max_tokens updated] {max_tokens}")
                        else:
                            history.append("[Error] Invalid value for max_tokens (expect integer)")
                        redraw()
                        current_input = ""
                        continue
                    elif prompt.lower().startswith("/temperature"):
                        value = prompt[len("/temperature"):].strip()
                        try:
                            temperature = float(value)
                            history.append(f"[temperature updated] {temperature}")
                        except ValueError:
                            history.append("[Error] Invalid value for temperature (expect float)")
                        redraw()
                        current_input = ""
                        continue
                    # Prepare context
                    messages = _build_messages(conversation, prompt)
                    ctx_tokens = sum(_estimate_tokens(m["content"]) for m in messages)

                    history.append(f"You: {prompt}")
                    redraw()
                    history.append("Model: ")
                    answer_idx = len(history) - 1
                    redraw()

                    start_time = time.time()
                    tokens_seen = 0
                    assistant_reply = ""
                    for token in _stream_vllm_messages(messages):
                        tokens_seen += len(token)
                        assistant_reply += token
                        history[answer_idx] += token
                        if auto_scroll:
                            scroll = max(0, total_lines - chat_height)
                        elapsed = max(time.time() - start_time, 1e-3)
                        tps = tokens_seen / elapsed
                        input_win.addstr(0, 2, f"TPS: {tps:.1f} | Ctx: {ctx_tokens}".ljust(max_x - 4))
                        input_win.noutrefresh()
                        curses.doupdate()

                        # Throttle screen refresh to at most ~20 FPS (50ms)
                        now = time.time()
                        if now - last_redraw >= 0.05:
                            redraw()
                            last_redraw = now

                    # Update conversation history for next turn
                    conversation.append(("user", prompt))
                    conversation.append(("assistant", assistant_reply))
                    # Final redraw to ensure full answer visible
                    redraw()
                # After completing an answer, re-enable auto-scroll
                auto_scroll = True
                current_input = ""
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                current_input = current_input[:-1]
            elif ch == curses.KEY_PPAGE:
                scroll = max(0, scroll - chat_height)
                auto_scroll = False  # User is manually scrolling
                redraw()
            elif ch == curses.KEY_NPAGE:
                bottom = max(0, total_lines - chat_height)
                scroll = min(bottom, scroll + chat_height)
                auto_scroll = (scroll == bottom)
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

def query_vllm_once(prompt: str, model_name: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
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

def query_vllm_stream(prompt: str, model_name: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
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
    parser.add_argument("--max-context", type=int, default=2048, help="Maximum approximate tokens to include in the context when chatting.")
    parser.add_argument("--system-prompt", "-s", type=str, default=DEFAULT_SYSTEM_PROMPT, help="Custom system prompt for the model.")
    
    args = parser.parse_args()

    model_name = get_first_available_model()
    
    if not model_name:
        sys.exit(1) # Exit the script if no model is fetched

        # Interactive chat
    if args.interactive or args.prompt is None:
        chat_cli(model_name, args.max_context, args.system_prompt)
        return

    print(f"💬 Sending question: \"{args.prompt}\"")

    if args.stream:
        query_vllm_stream(args.prompt, model_name, args.system_prompt)
    else:
        query_vllm_once(args.prompt, model_name, args.system_prompt)

if __name__ == "__main__":
    main()