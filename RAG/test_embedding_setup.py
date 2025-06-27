#!/usr/bin/env python3
"""
測試 embedding 模型設置是否正確的腳本
"""

import logging
import torch
from pathlib import Path
import os

# 設置日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_embedding_setup():
    """測試 embedding 模型設置是否正確"""
    
    print("🔍 測試 embedding 模型設置...")
    print("=" * 60)
    
    # 檢查專案目錄下的模型目錄
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    cache_dir = models_dir / "cache"
    embedding_dir = models_dir / "embedding"
    
    print(f"📁 專案目錄: {base_dir}")
    print(f"📁 模型目錄: {models_dir}")
    print(f"📁 緩存目錄: {cache_dir}")
    print(f"📁 Embedding 目錄: {embedding_dir}")
    
    # 測試環境變數設置
    print("\n🔧 測試環境變數設置...")
    try:
        # 導入 rag_api 來觸發環境變數設置
        import rag_api
        
        print("✅ rag_api 導入成功")
        print(f"HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
        print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")
        print(f"TORCH_HOME: {os.environ.get('TORCH_HOME', 'Not set')}")
        print(f"SENTENCE_TRANSFORMERS_HOME: {os.environ.get('SENTENCE_TRANSFORMERS_HOME', 'Not set')}")
        
    except Exception as e:
        print(f"❌ rag_api 導入失敗: {e}")
        return False
    
    # 測試目錄創建
    print("\n📁 檢查目錄創建...")
    if models_dir.exists():
        print("✅ models 目錄存在")
    else:
        print("❌ models 目錄不存在")
        return False
        
    if cache_dir.exists():
        print("✅ cache 目錄存在")
    else:
        print("❌ cache 目錄不存在")
        return False
        
    if embedding_dir.exists():
        print("✅ embedding 目錄存在")
    else:
        print("❌ embedding 目錄不存在")
        return False
    
    # 測試模型檢查函數
    print("\n🧪 測試模型檢查函數...")
    try:
        from rag_api import check_and_download_embedding_model, EMBED_MODEL_NAME
        
        print(f"🔍 檢查模型: {EMBED_MODEL_NAME}")
        result = check_and_download_embedding_model(EMBED_MODEL_NAME)
        
        if result:
            print("✅ 模型檢查成功")
        else:
            print("❌ 模型檢查失敗")
            return False
            
    except Exception as e:
        print(f"❌ 模型檢查函數測試失敗: {e}")
        return False
    
    # 測試 start_web.py 的初始化
    print("\n🚀 測試 start_web.py 初始化...")
    try:
        # 模擬 start_web.py 的初始化過程
        import rag_api
        from rag_api import check_and_download_embedding_model
        
        if check_and_download_embedding_model():
            print("✅ start_web.py 初始化檢查成功")
        else:
            print("❌ start_web.py 初始化檢查失敗")
            return False
            
    except Exception as e:
        print(f"❌ start_web.py 初始化測試失敗: {e}")
        return False
    
    print("\n🎉 所有測試通過！")
    print("✅ Embedding 模型設置正確")
    print("✅ 模型將下載到專案目錄")
    print("✅ start_web.py 初始化檢查正常")
    
    return True

def main():
    """主函數"""
    print("🔧 Embedding 模型設置測試工具")
    print("=" * 60)
    
    success = test_embedding_setup()
    
    if success:
        print("\n🎊 恭喜！所有測試都通過了！")
        print("\n📋 現在您可以:")
        print("  1. 運行: python start_web.py")
        print("  2. 訪問: http://127.0.0.1:8000/ui")
        print("  3. 上傳文件並建立索引")
        
        print("\n💡 提示:")
        print("  - 首次運行會自動下載 embedding 模型到專案目錄")
        print("  - 模型下載可能需要一些時間")
        print("  - 確保網路連接正常")
    else:
        print("\n❌ 測試失敗")
        print("請檢查錯誤信息並修復問題")

if __name__ == "__main__":
    main()