# RAG Web UI 使用說明

## 問題解決

如果您遇到 `{"detail":"Not Found"}` 錯誤，請按照以下步驟操作：

## 快速啟動步驟

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 初始化演示環境 (包含 embedding 模型檢查)
```bash
cd RAG
python chroma_llamaindex_example.py  # 建立基本索引並下載 embedding 模型
```

**注意**: 
- 首次運行會自動下載 embedding 模型到專案目錄 `RAG/models/embedding/`
- 模型下載可能需要幾分鐘時間，請耐心等待
- 模型將保存在專案目錄中，不會使用系統預設路徑

### 3. 啟動 Web UI
```bash
python start_web.py  # 啟動時會自動檢查 embedding 模型
```

**注意**: 
- `start_web.py` 在啟動時會自動檢查 embedding 模型是否存在
- 如果模型不存在，會自動下載到專案目錄
- 首次啟動可能需要額外時間來下載模型

### 4. 訪問 Web UI
打開瀏覽器訪問：
- 主頁面: http://127.0.0.1:8000/ui
- 健康檢查: http://127.0.0.1:8000/health
- API 文檔: http://127.0.0.1:8000/docs

## 替代啟動方法

### 方法 1: 使用 rag_web.py
```bash
python rag_web.py
```

### 方法 2: 使用 uvicorn 直接啟動
```bash
uvicorn rag_web:app --host 0.0.0.0 --port 8000 --reload
```

## 功能說明

1. **上傳檔案**: 支援 .txt, .pdf, .doc, .docx 格式
2. **建立索引**: 將上傳的檔案建立向量索引
3. **查詢**: 使用自然語言查詢已索引的內容
4. **狀態監控**: 查看檔案的索引狀態

## 故障排除

### Embedding 模型錯誤 (最常見)
如果遇到 `config_sentence_transformers.json` 錯誤：
```bash
cd RAG
python test_embedding_setup.py  # 測試 embedding 模型設置
# 或者
python start_web.py  # 啟動時會自動檢查和下載模型
```

**新功能**: 現在所有 embedding 模型都會自動下載到專案目錄 `RAG/models/` 中，不再使用系統預設路徑。

### 如果遇到導入錯誤
確保已安裝所有必要的套件：
```bash
pip install --upgrade llama-index llama-index-core llama-index-embeddings-huggingface llama-index-vector-stores-chroma sentence-transformers
```

### 如果 /ui 路由不存在
1. 檢查是否正確導入了 rag_web 模組
2. 確保 templates/index.html 文件存在
3. 使用 start_web.py 啟動以獲得更詳細的調試信息

### 如果查詢功能不工作
1. 先運行 `python init_demo.py` 建立基本索引
2. 或者上傳檔案並建立索引後再查詢

### 清除模型緩存 (如果模型損壞)
```bash
# 清除專案目錄中的模型 (推薦)
cd RAG
rm -rf models/  # Linux/Mac
rmdir /s models  # Windows

# 或清除系統預設緩存 (如果仍有問題)
rm -rf ~/.cache/huggingface/  # Linux/Mac
rmdir /s "%USERPROFILE%\.cache\huggingface"  # Windows
```

**注意**: 現在模型主要保存在專案目錄 `RAG/models/` 中，通常只需要清除專案目錄即可。

## 目錄結構
```
RAG/
├── rag_api.py              # API 後端 (包含 embedding 模型自動下載)
├── rag_web.py              # Web UI 路由
├── start_web.py            # 啟動腳本 (啟動時檢查模型)
├── test_embedding_setup.py   # Embedding 模型設置測試腳本
├── chroma_llamaindex_example.py  # 索引建立範例
├── requirements.txt        # 依賴套件
├── templates/
│   └── index.html         # Web UI 模板
├── uploads/               # 上傳檔案目錄
├── chroma_db/            # 向量資料庫
└── models/               # 模型目錄 (新增)
    ├── cache/           # HuggingFace 緩存
    └── embedding/       # Embedding 模型
```

**新增功能**: 
- 所有模型現在保存在專案目錄 `models/` 中
- 自動創建必要的目錄結構
- 啟動時自動檢查和下載 embedding 模型