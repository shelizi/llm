# RAG AI API 配置指南

## 功能概述

RAG 系統現在支援配置 OpenAI 相容的 API 來提供智能回答功能。配置後，系統將使用 AI 模型根據檢索到的文檔內容生成更智能、更自然的回答。

## 支援的 API 服務

### 1. OpenAI 官方 API
- **API 網址**: `https://api.openai.com`
- **模型**: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo` 等
- **Token 格式**: `sk-xxxxxxxxxx`

### 2. Azure OpenAI
- **API 網址**: `https://your-resource.openai.azure.com`
- **模型**: 您部署的模型名稱
- **Token 格式**: Azure API Key

### 3. 本地部署的相容 API
- **API 網址**: 您的本地服務地址，如 `http://localhost:8000`
- **模型**: 根據您的部署配置
- **Token 格式**: 根據您的服務要求

### 4. 其他 OpenAI 相容服務
- Anthropic Claude API (通過相容層)
- 各種開源模型的 API 服務

## 配置步驟

### 1. 訪問配置頁面
1. 啟動 RAG Web UI
2. 在主頁面找到 "⚙️ AI API 設定" 區域
3. 點擊 "設定" 按鈕展開配置表單

### 2. 填寫配置信息
1. **API 網址**: 輸入您的 API 服務地址
2. **模型名稱**: 輸入要使用的模型名稱
3. **API Token**: 輸入您的 API 金鑰

### 3. 測試連接
1. 填寫完配置信息後，點擊 "測試連接" 按鈕
2. 系統會發送測試請求驗證配置是否正確
3. 確認測試成功後再保存配置

### 4. 保存配置
1. 點擊 "保存配置" 按鈕
2. 配置將被加密存儲到服務器本地
3. 保存成功後，狀態徽章會變為 "已配置"

## 安全性說明

### 加密存儲
- API Token 使用 AES 加密算法安全存儲
- 加密金鑰存儲在服務器本地，僅所有者可讀取
- 配置文件權限設為 600（僅所有者可讀寫）

### 數據傳輸
- 所有 API 請求使用 HTTPS 加密傳輸
- Token 不會在前端頁面中明文顯示
- 日誌中不會記錄敏感信息

## 使用效果

### 配置前
- 查詢結果僅顯示檢索到的原始文檔片段
- 回答可能不夠自然或連貫

### 配置後
- 系統會使用 AI 模型基於檢索內容生成智能回答
- 回答更加自然、連貫、易理解
- 仍保留原始檢索結果作為參考

## 配置示例

### OpenAI API 配置
```
API 網址: https://api.openai.com
模型名稱: gpt-3.5-turbo
API Token: sk-proj-xxxxxxxxxxxxxxxxxxxx
```

### Azure OpenAI 配置
```
API 網址: https://your-resource.openai.azure.com
模型名稱: gpt-35-turbo
API Token: your-azure-api-key
```

### 本地 API 配置
```
API 網址: http://localhost:11434
模型名稱: llama2
API Token: your-local-token
```

## 故障排除

### 常見問題

1. **連接測試失敗**
   - 檢查 API 網址是否正確
   - 確認 Token 是否有效
   - 檢查網絡連接

2. **模型不存在錯誤**
   - 確認模型名稱拼寫正確
   - 檢查您的 API 服務是否支援該模型

3. **權限錯誤**
   - 確認 API Token 有足夠的權限
   - 檢查 API 服務的使用限制

4. **查詢時仍顯示原始回答**
   - 確認 API 配置狀態為 "已配置"
   - 檢查服務器日誌中的錯誤信息

### 日誌檢查
查看服務器日誌以獲取詳細的錯誤信息：
```bash
# 如果使用 Docker
docker logs rag-container

# 如果直接運行
查看控制台輸出
```

## 管理配置

### 修改配置
1. 展開 API 設定區域
2. 修改相應字段
3. 重新測試並保存

### 刪除配置
1. 點擊 "刪除配置" 按鈕
2. 確認刪除操作
3. 系統將回到僅使用檢索回答的模式

## 最佳實踐

1. **選擇合適的模型**
   - 對於中文內容，推薦使用支援中文的模型
   - 考慮成本和性能的平衡

2. **定期檢查配置**
   - 確保 API Token 未過期
   - 監控 API 使用量和成本

3. **備份配置**
   - 記錄您的配置信息（除了 Token）
   - 準備備用的 API 服務

4. **安全管理**
   - 定期更換 API Token
   - 限制 API Token 的權限範圍
   - 監控異常的 API 使用