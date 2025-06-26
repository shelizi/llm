# 本地模型配置指南

## 支援的本地模型服務

RAG 系統完全支援本地模型，無需 API Token！以下是常見的本地模型服務配置：

### 1. Ollama

Ollama 是最受歡迎的本地模型運行平台。

#### 安裝 Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# 下載並安裝 https://ollama.ai/download
```

#### 啟動模型
```bash
# 下載並運行模型
ollama run llama2
ollama run codellama
ollama run mistral
ollama run llama2-chinese  # 中文模型
```

#### RAG 配置
```
API 網址: http://localhost:11434
模型名稱: llama2 (或其他已下載的模型)
API Token: (留空)
```

### 2. LocalAI

LocalAI 是另一個優秀的本地 AI 服務。

#### 使用 Docker 啟動
```bash
docker run -p 8080:8080 --name local-ai -ti localai/localai:latest
```

#### RAG 配置
```
API 網址: http://localhost:8080
模型名稱: gpt-3.5-turbo (或配置的模型名)
API Token: (留空)
```

### 3. Text Generation WebUI

支援多種開源模型的 Web 界面。

#### RAG 配置
```
API 網址: http://localhost:5000
模型名稱: 根據載入的模型
API Token: (留空)
```

### 4. vLLM

高性能的推理服務器。

#### 啟動服務
```bash
python -m vllm.entrypoints.openai.api_server \
  --model microsoft/DialoGPT-medium \
  --port 8000
```

#### RAG 配置
```
API 網址: http://localhost:8000
模型名稱: microsoft/DialoGPT-medium
API Token: (留空)
```

## 配置步驟

### 1. 啟動本地模型服務
選擇上述任一服務並啟動。

### 2. 在 RAG Web UI 中配置
1. 訪問 http://localhost:8001/ui
2. 找到 "⚙️ AI API 設定" 區域
3. 點擊 "設定" 按鈕
4. 填寫配置：
   - **API 網址**: 本地服務地址（如 http://localhost:11434）
   - **模型名稱**: 模型名稱（如 llama2）
   - **API Token**: **留空**（本地模型不需要）

### 3. 測試連接
點擊 "測試連接" 確認配置正確。

### 4. 保存配置
測試成功後點擊 "保存配置"。

## 常見配置範例

### Ollama + Llama2
```
API 網址: http://localhost:11434
模型名稱: llama2
API Token: (留空)
```

### Ollama + 中文模型
```
API 網址: http://localhost:11434
模型名稱: qwen:7b
API Token: (留空)
```

### LocalAI
```
API 網址: http://localhost:8080
模型名稱: gpt-3.5-turbo
API Token: (留空)
```

### 自定義端口的 Ollama
```
API 網址: http://localhost:11435
模型名稱: codellama
API Token: (留空)
```

## 推薦的中文模型

### 1. Qwen (通義千問)
```bash
ollama run qwen:7b
ollama run qwen:14b
```

### 2. ChatGLM
```bash
ollama run chatglm3:6b
```

### 3. Baichuan
```bash
ollama run baichuan2:7b
```

### 4. Yi (零一萬物)
```bash
ollama run yi:6b
```

## 性能優化建議

### 1. 硬體需求
- **RAM**: 至少 8GB，推薦 16GB+
- **GPU**: NVIDIA GPU 可大幅提升速度
- **存儲**: SSD 推薦

### 2. Ollama GPU 加速
```bash
# 確認 GPU 支援
ollama run llama2 --gpu

# 設定 GPU 記憶體限制
OLLAMA_GPU_MEMORY_LIMIT=4GB ollama run llama2
```

### 3. 調整模型參數
在 RAG 系統中，可以通過修改 `openai_client.py` 調整：
```python
request_data = {
    "model": model_name,
    "messages": messages,
    "temperature": 0.3,  # 降低隨機性
    "max_tokens": 500,   # 限制回答長度
    "top_p": 0.9,       # 控制多樣性
}
```

## 故障排除

### 1. 連接失敗
- 確認本地服務正在運行
- 檢查端口是否正確
- 確認防火牆設定

### 2. 模型未找到
- 確認模型已正確下載
- 檢查模型名稱拼寫
- 查看服務日誌

### 3. 回應緩慢
- 考慮使用較小的模型
- 啟用 GPU 加速
- 調整 max_tokens 參數

### 4. 記憶體不足
```bash
# Ollama 限制記憶體使用
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MEMORY_LIMIT=4GB ollama serve
```

## 與雲端 API 的比較

| 特性 | 本地模型 | 雲端 API |
|------|----------|----------|
| 成本 | 免費（除硬體） | 按使用量付費 |
| 隱私 | 完全私密 | 數據上傳到雲端 |
| 速度 | 取決於硬體 | 通常較快 |
| 模型選擇 | 有限 | 豐富 |
| 設定複雜度 | 較高 | 較低 |
| 離線使用 | 支援 | 不支援 |

## 安全性說明

### 本地模型的優勢
- **數據隱私**: 所有數據在本地處理，不會上傳到雲端
- **無需 API Key**: 不用擔心 API 金鑰洩露
- **完全控制**: 對模型和數據有完全控制權

### 注意事項
- 確保本地服務僅在受信任的網絡中運行
- 定期更新模型和服務軟體
- 備份重要的模型和配置

## 進階配置

### 1. 多模型切換
可以配置多個本地模型，在不同場景下使用：
- 文檔問答：使用通用模型（如 llama2）
- 程式碼分析：使用程式碼模型（如 codellama）
- 中文內容：使用中文優化模型（如 qwen）

### 2. 負載均衡
如果有多台機器，可以配置負載均衡：
```
API 網址: http://load-balancer:8080
```

### 3. 自定義 Prompt
修改 `openai_client.py` 中的 system prompt 以適應特定需求。

## 社群資源

- [Ollama 官方文檔](https://ollama.ai/docs)
- [LocalAI GitHub](https://github.com/go-skynet/LocalAI)
- [Awesome Local AI](https://github.com/janhq/awesome-local-ai)
- [模型排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)