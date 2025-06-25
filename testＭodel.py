import requests
import argparse
import json
import sys

# --- 設定 ---
# 您的 vLLM OpenAI API 伺服器運行的位址
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
# 獲取模型列表的 API 位址
VLLM_MODELS_URL = "http://localhost:8000/v1/models"
HEADERS = {"Content-Type": "application/json"}

def get_first_available_model():
    """自動從 vLLM 伺服器獲取第一個可用的模型名稱"""
    try:
        response = requests.get(VLLM_MODELS_URL)
        response.raise_for_status()  # 如果請求失敗則拋出異常
        models = response.json()
        if "data" in models and len(models["data"]) > 0:
            model_name = models["data"][0]["id"]
            print(f"✅ 自動偵測到模型: {model_name}")
            return model_name
        else:
            print("❌ 錯誤：無法從伺服器獲取模型列表，請檢查伺服器是否已正確加載模型。")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 錯誤：無法連接到 vLLM 伺服器 at {VLLM_MODELS_URL}。")
        print("請確認：")
        print("1. vLLM 伺服器是否已成功啟動？")
        print("2. 伺服器位址和埠號是否正確？")
        print(f"詳細錯誤: {e}")
        return None

def query_vllm_once(prompt: str, model_name: str):
    """發送請求並一次性獲取完整回覆"""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": False # 設為 False
    }

    try:
        response = requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        
        data = response.json()
        content = data['choices'][0]['message']['content']

        print("\n--- vLLM 回應 (一次性) ---")
        print(content)
        print("--------------------------\n")

    except requests.exceptions.RequestException as e:
        print(f"❌ 請求失敗: {e}")

def query_vllm_stream(prompt: str, model_name: str):
    """發送請求並以串流方式處理回覆"""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": True # 設為 True
    }

    try:
        print("\n--- vLLM 回應 (串流) ---")
        with requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload), stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    # vLLM 的串流輸出以 "data: " 開頭
                    if decoded_chunk.startswith('data: '):
                        json_str = decoded_chunk[len('data: '):]
                        if json_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(json_str)
                            delta = data['choices'][0]['delta']
                            if 'content' in delta:
                                content = delta['content']
                                # 使用 end='' 和 flush=True 實現即時打印
                                print(content, end='', flush=True)
                        except json.JSONDecodeError:
                            print(f"\n無法解析的 JSON 片段: {json_str}")
        print("\n--------------------------\n")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ 請求失敗: {e}")

def main():
    parser = argparse.ArgumentParser(description="一個用來測試 vLLM OpenAI API 的 Python 腳本。")
    parser.add_argument("prompt", type=str, help="您要向大型語言模型提問的問題。")
    parser.add_argument("--stream", action="store_true", help="使用串流模式接收回覆。")
    
    args = parser.parse_args()

    model_name = get_first_available_model()
    
    if not model_name:
        sys.exit(1) # 如果沒有獲取到模型，則退出腳本

    print(f"💬 正在發送問題: \"{args.prompt}\"")

    if args.stream:
        query_vllm_stream(args.prompt, model_name)
    else:
        query_vllm_once(args.prompt, model_name)

if __name__ == "__main__":
    main()