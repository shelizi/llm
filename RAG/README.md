# RAG 智能文檔問答系統

> 一個功能完整的 RAG (Retrieval-Augmented Generation) 系統，支援文檔上傳、向量索引建立、智能查詢，並整合 AI API 進行增強回答生成。

## 🌟 主要功能

### 📄 文檔處理
- ✅ **多格式支援**: PDF、TXT、DOC、DOCX
- ✅ **批量上傳**: 支援多檔案同時處理
- ✅ **智能分割**: 可調整 Chunk Size 和 Overlap 參數
- ✅ **狀態追蹤**: 即時顯示索引建立進度

### 🔍 智能查詢
- ✅ **向量檢索**: 基於語義相似度的文檔搜尋
- ✅ **檔案過濾**: 可選擇查詢特定文檔或所有文檔
- ✅ **結果調整**: 可設定返回 1-20 個相關文檔
- ✅ **相似度評分**: 顯示檢索結果的相關性分數

### 🤖 AI 增強回答
- ✅ **OpenAI 相容 API**: 支援多種 AI 服務
- ✅ **本地模型支援**: 完美支援 Ollama、LocalAI 等
- ✅ **加密存儲**: API Token 使用 AES 加密安全存儲
- ✅ **智能回答**: 基於檢索內容生成自然語言回答

### 🏠 本地模型優勢
- 🛡️ **完全隱私**: 數據不離開本地環境
- 🔐 **無需 API Key**: 避免金鑰洩露風險
- 💰 **零成本**: 除硬體外完全免費
- 📱 **離線運行**: 無網絡也能使用

## 🚀 快速開始

### 1. 安裝依賴

```bash
cd RAG
pip install -r requirements.txt
```

**主要依賴:**
- `llama-index` - 向量索引和檢索
- `chromadb` - 向量資料庫
- `fastapi` - Web API 框架
- `cryptography` - 配置加密
- `aiohttp` - HTTP 客戶端

### 2. 啟動服務

```bash
python start_web.py
```

### 3. 訪問 Web UI

- **主頁面**: http://localhost:8001/ui
- **API 文檔**: http://localhost:8001/docs
- **健康檢查**: http://localhost:8001/health

## 📋 使用指南

### 文檔上傳與索引

1. **上傳檔案**
   - 在 "上傳檔案" 區域選擇文檔
   - 支援 PDF、TXT、DOC、DOCX 格式
   - 檔案會自動保存到 `uploads/` 目錄

2. **建立索引**
   - 在檔案狀態表格中勾選要處理的檔案
   - 調整索引參數（可選）:
     - **Chunk Size**: 文本分割大小（預設 512）
     - **Overlap**: 重疊字符數（預設 50）
   - 點擊 "開始索引選中的檔案"

3. **狀態監控**
   - **未索引**: 檔案已上傳但未處理
   - **索引中**: 正在建立索引
   - **已索引**: 可用於查詢
   - **錯誤**: 處理失敗，可重試

### AI API 配置

#### 🏠 本地模型配置（推薦）

**Ollama 配置:**
```
API 網址: http://localhost:11434
模型名稱: llama2
API Token: (留空 - 本地模型不需要)
```

**LocalAI 配置:**
```
API 網址: http://localhost:8080
模型名稱: gpt-3.5-turbo
API Token: (留空 - 本地模型不需要)
```

#### ☁️ 雲端服務配置

**OpenAI API:**
```
API 網址: https://api.openai.com
模型名稱: gpt-3.5-turbo
API Token: sk-proj-xxxxxxxxxxxxxxxxxxxx
```

**Azure OpenAI:**
```
API 網址: https://your-resource.openai.azure.com
模型名稱: gpt-35-turbo
API Token: your-azure-api-key
```

#### 配置步驟

1. **訪問配置頁面**
   - 在主頁面找到 "⚙️ AI API 設定" 區域
   - 點擊 "設定" 按鈕展開配置表單

2. **填寫配置信息**
   - **API 網址**: 輸入服務地址
   - **模型名稱**: 輸入模型名稱
   - **API Token**: 輸入金鑰（本地模型可留空）

3. **測試並保存**
   - 點擊 "測試連接" 驗證配置
   - 測試成功後點擊 "保存配置"
   - 配置將被加密存儲

### 智能查詢

1. **基本查詢**
   - 在 "智能查詢" 區域輸入問題
   - 設定返回結果數量（1-20個）
   - 點擊 "查詢" 或按回車鍵

2. **檔案過濾**
   - **所有文檔**: 查詢整個資料庫
   - **指定文檔**: 選擇特定檔案進行查詢
   - 使用 "全選/清除" 快速操作

3. **結果解讀**
   - **AI 智能回答**: 基於檢索內容的 AI 生成回答
   - **參考來源**: 顯示相關文檔片段和相似度分數
   - **查詢統計**: 顯示檢索結果數量和最高相似度

## 🏠 本地模型設置

### Ollama 安裝與配置

#### 安裝 Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# 下載並安裝 https://ollama.ai/download
```

#### 下載模型
```bash
# 通用模型
ollama run llama2
ollama run codellama
ollama run mistral

# 中文優化模型
ollama run qwen:7b
ollama run chatglm3:6b
ollama run baichuan2:7b
```

#### RAG 配置
```
API 網址: http://localhost:11434
模型名稱: llama2 (或其他已下載的模型)
API Token: (完全留空)
```

### LocalAI 設置

#### Docker 啟動
```bash
docker run -p 8080:8080 --name local-ai -ti localai/localai:latest
```

#### RAG 配置
```
API 網址: http://localhost:8080
模型名稱: gpt-3.5-turbo
API Token: (完全留空)
```

### 推薦中文模型

1. **Qwen (通義千問)**
   ```bash
   ollama run qwen:7b
   ollama run qwen:14b
   ```

2. **ChatGLM**
   ```bash
   ollama run chatglm3:6b
   ```

3. **Baichuan**
   ```bash
   ollama run baichuan2:7b
   ```

4. **Yi (零一萬物)**
   ```bash
   ollama run yi:6b
   ```

## 🔒 安全性說明

### 加密存儲
- API Token 使用 **AES-256 加密**安全存儲
- 加密金鑰存儲在服務器本地，僅所有者可讀取
- 配置文件權限設為 **600**（僅所有者可讀寫）

### 數據隱私
- **本地處理**: 使用本地模型時，所有數據在本地處理
- **HTTPS 傳輸**: 雲端 API 請求使用加密傳輸
- **無敏感日誌**: 系統不會記錄 API Token 等敏感信息

### 權限管理
- 配置文件和金鑰文件設置嚴格權限
- 前端不會顯示完整的 API Token
- 支援配置的安全刪除功能

## 📊 API 文檔

### 查詢端點

**POST /query**

請求格式:
```json
{
  "query": "您的問題",
  "top_k": 5,
  "filenames": ["可選的檔案列表"]
}
```

回應格式:
```json
{
  "answer": "檢索回答",
  "enhanced_answer": "AI增強回答",
  "answer_source": "ai_enhanced",
  "sources": [
    {
      "filename": "檔案名稱",
      "snippet": "相關片段",
      "score": 0.85
    }
  ]
}
```

### 配置端點

**GET /api_config_status** - 獲取 API 配置狀態
**POST /save_api_config** - 保存 API 配置
**POST /test_api_config** - 測試 API 連接
**POST /delete_api_config** - 刪除 API 配置

### 檔案管理端點

**GET /indexed_files** - 獲取已索引檔案列表
**POST /upload** - 上傳檔案
**POST /index** - 建立索引
**GET /files** - 獲取檔案列表及狀態

## 🛠️ 故障排除

### 常見問題

1. **Embedding 模型錯誤**
   ```bash
   # 測試模型設置
   python test_embedding_setup.py
   
   # 清除損壞的模型
   rm -rf models/
   ```

2. **API 連接失敗**
   - 檢查 API 網址是否正確
   - 確認 Token 是否有效（雲端服務）
   - 檢查本地服務是否運行（本地模型）

3. **查詢無結果**
   - 確認檔案已正確索引
   - 嘗試調整查詢關鍵詞
   - 增加 top_k 值

4. **本地模型連接問題**
   - 確認 Ollama/LocalAI 服務正在運行
   - 檢查端口是否正確（11434/8080）
   - 確認模型已下載

### 性能優化

1. **硬體建議**
   - **RAM**: 至少 8GB，推薦 16GB+
   - **GPU**: NVIDIA GPU 可大幅提升本地模型速度
   - **存儲**: SSD 推薦

2. **模型選擇**
   - **快速回應**: 使用較小模型（7B）
   - **質量優先**: 使用較大模型（13B+）
   - **中文內容**: 選擇中文優化模型

3. **參數調整**
   - **Chunk Size**: 較小值適合精確查詢，較大值適合上下文理解
   - **Overlap**: 增加重疊可提高連續性，但會增加存儲需求
   - **Top K**: 較大值提供更全面結果，但處理時間較長

## 📁 目錄結構

```
RAG/
├── README.md                    # 本說明文檔
├── requirements.txt             # Python 依賴
├── start_web.py                # 啟動腳本
├── rag_api.py                  # API 後端
├── rag_web.py                  # Web UI 路由
├── config_manager.py           # 配置管理（加密）
├── openai_client.py            # AI API 客戶端
├── test_api_config.py          # API 配置測試
├── test_local_model.py         # 本地模型測試
├── templates/
│   └── index.html              # Web UI 模板
├── uploads/                    # 上傳檔案目錄
├── chroma_db/                  # 向量資料庫
├── config/                     # 加密配置目錄
│   ├── config.key              # 加密金鑰
│   └── api_config.enc          # 加密的 API 配置
└── models/                     # 模型目錄
    ├── cache/                  # HuggingFace 緩存
    └── embedding/              # Embedding 模型
```

## 🧪 測試功能

### 運行測試

```bash
# 測試 API 配置功能
python test_api_config.py

# 測試本地模型支援
python test_local_model.py

# 測試 Embedding 模型
python test_embedding_setup.py
```

### 測試覆蓋

- ✅ **配置管理**: 加密存儲和讀取
- ✅ **API 連接**: 各種服務的連接測試
- ✅ **檔案處理**: 上傳、索引、查詢流程
- ✅ **本地模型**: 無 Token 配置和使用
- ✅ **錯誤處理**: 各種異常情況的處理

## 🔄 版本更新

### 最新功能 (v2.0)

- 🆕 **AI API 整合**: 支援 OpenAI 相容 API
- 🆕 **本地模型支援**: 完美支援 Ollama、LocalAI
- 🆕 **加密配置**: API Token 安全存儲
- 🆕 **檔案過濾**: 可選擇查詢特定檔案
- 🆕 **增強 UI**: 現代化的查詢界面
- 🆕 **智能回答**: AI 生成的自然語言回答

### 向後兼容

- ✅ 所有現有 API 保持兼容
- ✅ 現有檔案和索引無需重建
- ✅ 配置檔案自動遷移

## 🤝 貢獻指南

### 開發環境設置

1. Fork 本專案
2. 創建功能分支
3. 安裝開發依賴
4. 運行測試確保功能正常
5. 提交 Pull Request

### 程式碼規範

- 使用 Python 3.8+
- 遵循 PEP 8 程式碼風格
- 添加適當的註釋和文檔
- 確保測試覆蓋率

## 📞 支援與社群

### 獲取幫助

1. **檢查文檔**: 首先查看本 README 和相關指南
2. **運行測試**: 使用提供的測試腳本診斷問題
3. **查看日誌**: 檢查控制台輸出和錯誤信息
4. **社群討論**: 在 GitHub Issues 中提問

### 常用資源

- [Ollama 官方文檔](https://ollama.ai/docs)
- [LlamaIndex 文檔](https://docs.llamaindex.ai/)
- [ChromaDB 文檔](https://docs.trychroma.com/)
- [FastAPI 文檔](https://fastapi.tiangolo.com/)

## 📄 授權條款

本專案採用 MIT 授權條款，詳見 LICENSE 檔案。

---

**🎉 享受您的 RAG 智能文檔問答系統！**

> 如有任何問題或建議，歡迎提出 Issue 或 Pull Request。