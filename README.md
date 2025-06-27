# LLM 專案說明

## 目前支援的模型

以下列出 `models_list.txt` 中已整合、可直接使用的模型清單：

- Llama-3 Nemotron Super 49B
- Mistral-Small
- StarCode2-15B
- Phi-4-reasoning
- StarCode2-7B
- StarCode2-3B
- Devstral
- Gemma-3-27B
- Phi-4
- Codestral-22B
- Magistral

> 如需下載連結與檔案名稱，請參考 `models_list.txt`。

## 待新增章節

以下章節將於後續版本陸續補充，歡迎提出建議或 PR：


- Embedding Model 選項（本節內容已補充於下方）

## LLM WebUI 選項

- **OpenWebUI**：以其強大的可擴充性和基於管線（Pipeline）的架構脫穎而出，特別適合需要高度客製化工作流程與精細管理權限的場景。
- **LibreChat**：作為一個功能完備的 ChatGPT 替代品，其強項在於全面的企業級身份驗證整合與多樣化的後端支援，是大型組織部署的理想選擇。
- **Oobabooga's Text-Generation-WebUI**：在本地模型運行與實驗領域中堪稱翹楚，為高階使用者提供了無與倫比的生成參數控制與模型微調能力。
- **AnythingLLM**：專注於檢索增強生成（RAG），提供了一個一站式的解決方案，讓使用者能輕鬆地與自己的文件和知識庫進行對話。

## RAG 相關

- **FlowiseAI**：視覺化 Agent 與工作流程建構器。核心理念：FlowiseAI 是一個開源低程式碼平台，基於 LangChain 與 LlamaIndex，將複雜後端邏輯抽象為可拖放節點。
- **AnythingLLM**：企業級私有化 AI 應用裝置。一體化私有 RAG 與 AI Agent，重視多使用者支持、資料隱私與無程式碼易用性。
- **Verba**：Weaviate 原生 RAG 應用，由 Weaviate 團隊打造，展示向量資料庫強大功能並降低使用門檻。
- **LlamaIndex + Chainlit**：
  - **LlamaIndex**：數據框架，將客製化資料源連接到 LLM，提供最大靈活性與控制權。
  - **Chainlit**：專為建構生產級對話式 AI 應用的 Python 套件，與 LlamaIndex/LangChain 深度整合，可快速建立 UI。

## Embedding 模型選項

參考:https://huggingface.co/spaces/mteb/leaderboard

### 1. intfloat/multilingual-e5-large-instruct
**類型**：高效能通用型
- **開發團隊**：微軟研究團隊
- **基礎架構**：XLM-RoBERTa-large
- **參數量**：5.6億 (560M)
- **嵌入維度**：1024
- **最大序列長度**：512 tokens
- **語言支援**：100種語言（含中文）
- **特點**：
  - 在 MMTEB 基準測試中表現優異
  - 支援多語言檢索任務
  - 經過指令微調，適合零樣本檢索

### 2. nomic-ai/nomic-embed-text-v2
**類型**：高效率創新者
- **開發團隊**：Nomic AI
- **架構**：首創 MoE（專家混合模型）
- **總參數量**：4.75億（活躍參數3.05億）
- **嵌入維度**：768（可截斷至256）
- **最大序列長度**：512 tokens
- **語言支援**：多語言（含中文）
- **特點**：
  - 計算效率高
  - 在 MIRACL 基準測試表現優異
  - 開源訓練數據和代碼

### 3. jinaai/jina-embeddings-v2-base-zh
**類型**：長文本專家
- **開發團隊**：Jina AI
- **架構**：JinaBERT + ALiBi
- **參數量**：1.61億 (161M)
- **嵌入維度**：768
- **最大序列長度**：8192 tokens（顯著優於同類）
- **語言支援**：中英雙語
- **特點**：
  - 超長文本處理能力
  - 無需複雜分塊策略
  - 適合處理完整文檔或章節

### 4. Linq-AI-Research/Linq-Embed-Mistral
**類型**：高效能通用型
- **開發團隊**：Linq AI Research
- **基礎架構**：Mistral
- **參數量**：7B
- **嵌入維度**4096
- **最大序列長度**：32768 tokens
- **語言支援**：多語言（含中文）
- **特點**：
  - 在 MMTEB 基準測試中表現優異
  - 支援多語言檢索任務
  - 經過指令微調，適合零樣本檢索

## 向量資料庫選擇

- **LlamaIndex SimpleVectorStore**
  - 內建輕量級解決方案，適合快速原型開發與測試
  - 無需額外依賴，適合小型專案或學習用途

- **ChromaDB** ★ 最易上手推薦
  - 極簡 Docker 部署流程
  - 與 LlamaIndex 高度整合
  - 適合：快速原型開發、概念驗證 (PoC)

- **Qdrant** ★ 平衡易用性與擴展性
  - 單行 Docker 指令即可啟動
  - 基於 Rust 的高效能架構
  - 支援分散式部署
  - 適合：需要從開發平滑過渡到生產的專案

- **Weaviate** 功能豐富的企業級平台
  - 整合資料處理與 AI 模型推理
  - 模組化設計，高度可擴展
  - 適合：需要進階功能與企業級支援的大型應用

> 建議：初學者可從 ChromaDB 開始，待專案規模擴大後再評估是否遷移至 Qdrant 或 Weaviate。

