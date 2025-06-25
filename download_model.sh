#!/bin/bash

# Script to download various LLM models
SCRIPT_DIR="$(dirname "$0")"
MODELS_DIR="$SCRIPT_DIR/models"
MODELS_LIST="$SCRIPT_DIR/models_list.txt"

# 創建模型下載目錄
mkdir -p "$MODELS_DIR" 2>/dev/null

# 檢查模型列表檔案是否存在
if [ ! -f "$MODELS_LIST" ]; then
  echo "錯誤: 找不到模型列表檔案 '$MODELS_LIST'"
  exit 1
fi

# 讀取模型列表檔案
declare -a MODEL_NAMES
declare -a MODEL_FILES
declare -a MODEL_URLS

# 初始化計數器
index=1

# 讀取每一行，忽略以#開頭的註釋行
while IFS='|' read -r name file url || [ -n "$name" ]; do
  # 跳過註釋行
  [[ $name =~ ^#.* ]] && continue
  # 跳過空行
  [ -z "$name" ] && continue

  MODEL_NAMES[$index]="$name"
  MODEL_FILES[$index]="$file"
  
  # 檢查URL是否為檔案參考
  if [[ "$url" == file:* ]]; then
    # 提取檔案名稱
    url_file="${url#file:}"
    # 讀取檔案內容作為URL
    MODEL_URLS[$index]="$(cat "$SCRIPT_DIR/$url_file" 2>/dev/null)"
  else
    MODEL_URLS[$index]="$url"
  fi
  
  ((index++))
done < "$MODELS_LIST"

# 檢查是否有模型被成功讀取
if [ ${#MODEL_NAMES[@]} -eq 0 ]; then
  echo "錯誤: 未能從列表檔案讀取任何模型資訊"
  exit 1
fi

# 顯示模型選單
echo "===== LLM 模型下載工具 ====="
echo "請選擇一個模型下載:"
for i in "${!MODEL_NAMES[@]}"; do
  echo "$i. ${MODEL_NAMES[$i]} (${MODEL_FILES[$i]})"
done
echo "0. 退出"
echo "=============================" 

# 獲取用戶選擇
read -p "輸入您的選擇 (0-${#MODEL_NAMES[@]}): " choice

# 驗證輸入
if [[ ! "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 0 ] || [ "$choice" -gt "${#MODEL_NAMES[@]}" ]; then
  echo "無效選擇。請重試。"
  exit 1
fi

# 如果用戶選擇0，退出
if [ "$choice" -eq 0 ]; then
  echo "退出中..."
  exit 0
fi

# 準備下載所選模型
MODEL_NAME="${MODEL_NAMES[$choice]}"
OUTPUT_FILE="${MODEL_FILES[$choice]}"
URL="${MODEL_URLS[$choice]}"

# Ask if user wants background download
read -p "Do you want to run the download in background? (y/N): " bg_choice
if [[ "$bg_choice" =~ ^[Yy]$ ]]; then
  echo "Starting $MODEL_NAME download in background..."
  # Run with nohup so it continues if the terminal closes; write progress to a log
  nohup curl -L -o "$MODELS_DIR/$OUTPUT_FILE" "$URL" > "$MODELS_DIR/$OUTPUT_FILE.log" 2>&1 &
  pid=$!
  echo "Download PID: $pid"
  echo "Progress log: $MODELS_DIR/$OUTPUT_FILE.log"
  echo "You can run 'tail -f \"$MODELS_DIR/$OUTPUT_FILE.log\"' to watch progress or 'wait $pid' to wait in foreground."
  exit 0
fi

echo "正在下載 $MODEL_NAME ($OUTPUT_FILE)..."
echo "視您的網路連線和模型大小，這可能需要一些時間。"

# 執行下載（前景）
curl -L -o "$MODELS_DIR/$OUTPUT_FILE" "$URL"

if [ $? -eq 0 ]; then
  echo "下載成功完成！"
  echo "模型已儲存至: $MODELS_DIR/$OUTPUT_FILE"
else
  echo "下載失敗，錯誤代碼: $?"
fi
