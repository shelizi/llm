# RAG 系統 AI API 功能設置指南

## 新功能概述

RAG 系統現已支援 OpenAI 相容 API 進行智能回答生成！新增功能包括：

- ⚙️ **前端 API 配置**: 在 Web UI 中直接配置 OpenAI 相容 API
- 🔐 **加密存儲**: API Token 使用 AES 加密安全存儲
- 🤖 **AI 增強回答**: 基於檢索內容生成更智能的回答
- 🔌 **多服務支援**: 支援 OpenAI、Azure OpenAI、本地 API 等

## 安裝依賴

### 1. 安裝新增的 Python 套件

```bash
cd RAG
pip install cryptography aiohttp
```

或者使用更新的 requirements.txt：

```bash
pip install -r requirements.txt
```

### 2. 確認現有依賴

確保已安裝所有必要的依賴：

```bash
pip list | grep -E "(cryptography|aiohttp|fastapi|uvicorn)"
```

## 啟動服務

### 方法 1: 使用啟動腳本（推薦）

```bash
cd RAG
python start_web.py
```

### 方法 2: 直接啟動

```bash
cd RAG
uvicorn rag_web:app --host 0.0.0.0 --port 8001
```

## 訪問 Web UI

啟動後訪問：http://localhost:8001/ui

## 配置 AI API

### 1. 找到 API 設定區域
在 Web UI 主頁面中找到 "⚙️ AI API 設定" 區域，點擊 "設定" 按鈕展開配置表單。

### 2. 填寫 API 信息

#### OpenAI 官方 API
```
API 網址: https://api.openai.com
模型名稱: gpt-3.5-turbo
API Token: sk-proj-xxxxxxxxxxxxxxxxxx
```

#### Azure OpenAI
```
API 網址: https://your-resource.openai.azure.com
模型名稱: gpt-35-turbo
API Token: your-azure-api-key
```

#### 本地 Ollama
```
API 網址: http://localhost:11434
模型名稱: llama2
API Token: (可以留空或填入任意值)
```

### 3. 測試和保存
1. 點擊 "測試連接" 確認配置正確
2. 測試成功後點擊 "保存配置"
3. 狀態徽章會變為 "已配置"

## 使用 AI 增強回答

配置完成後：

1. **上傳文檔**: 在 "上傳檔案" 區域上傳 PDF、TXT、DOCX 文件
2. **建立索引**: 選擇文件並建立向量索引
3. **智能查詢**: 在 "智能查詢" 區域輸入問題
4. **獲得 AI 回答**: 系統會顯示基於文檔內容的 AI 生成回答

## 功能差異

### 配置前
- 僅顯示檢索到的原始文檔片段
- 回答可能不夠自然

### 配置後
- 顯示 AI 生成的智能回答
- 回答更自然、連貫、易理解
- 仍保留原始檢索結果作為參考

## 安全性說明

- API Token 使用 AES-256 加密存儲
- 配置文件權限限制為僅所有者可讀寫
- 前端不會顯示完整的 API Token
- 所有 API 通信使用 HTTPS 加密

## 故障排除

### 常見問題

1. **"cryptography" 模組未找到**
   ```bash
   pip install cryptography
   ```

2. **"aiohttp" 模組未找到**
   ```bash
   pip install aiohttp
   ```

3. **API 測試失敗**
   - 檢查網絡連接
   - 確認 API URL 和 Token 正確
   - 檢查 API 服務狀態

4. **查詢時仍顯示原始回答**
   - 確認 API 配置狀態為 "已配置"
   - 檢查服務器日誌中的錯誤信息

### 查看日誌

```bash
# 如果使用 start_web.py 啟動
python start_web.py

# 查看詳細日誌
python start_web.py --log-level debug
```

## 進階配置

### 自定義 System Prompt

如需修改 AI 回答的風格，可編輯 `openai_client.py` 中的 `_build_messages` 方法。

### 調整超時設置

在 `openai_client.py` 中修改 `ClientTimeout` 參數：

```python
timeout = aiohttp.ClientTimeout(total=120)  # 改為 120 秒
```

### 配置目錄

配置文件存儲在 `RAG/config/` 目錄下：
- `config.key`: 加密金鑰
- `api_config.enc`: 加密的 API 配置

## 更新和維護

### 更新配置
1. 在 Web UI 中修改 API 設定
2. 重新測試並保存

### 備份配置
```bash
# 備份配置目錄
cp -r RAG/config/ RAG/config_backup/
```

### 重置配置
如需重置所有配置：
```bash
rm -rf RAG/config/
```

## 支援的 API 服務

- ✅ OpenAI GPT-3.5/GPT-4
- ✅ Azure OpenAI
- ✅ Anthropic Claude (通過相容層)
- ✅ 本地 Ollama
- ✅ 其他 OpenAI 相容 API

## 聯繫和支援

如遇到問題，請：
1. 檢查本指南的故障排除部分
2. 查看服務器日誌
3. 確認所有依賴已正確安裝