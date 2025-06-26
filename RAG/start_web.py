#!/usr/bin/env python3
"""
啟動 RAG Web UI 服務器的簡化腳本
"""

import logging
import uvicorn
from pathlib import Path

# 設置日誌
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    try:
        # 確保必要的目錄存在
        base_dir = Path(__file__).parent
        upload_dir = base_dir / "uploads"
        templates_dir = base_dir / "templates"
        
        upload_dir.mkdir(exist_ok=True)
        templates_dir.mkdir(exist_ok=True)
        
        logging.info(f"基礎目錄: {base_dir}")
        logging.info(f"上傳目錄: {upload_dir} (存在: {upload_dir.exists()})")
        logging.info(f"模板目錄: {templates_dir} (存在: {templates_dir.exists()})")
        logging.info(f"index.html: {templates_dir / 'index.html'} (存在: {(templates_dir / 'index.html').exists()})")
        
        # 初始化檢查 embedding 模型
        logging.info("🔍 初始化檢查 embedding 模型...")
        try:
            # 先導入 rag_api 來設置環境變數和檢查模型
            import rag_api
            from rag_api import check_and_download_embedding_model
            
            # 檢查並下載 embedding 模型
            if check_and_download_embedding_model():
                logging.info("✅ Embedding 模型檢查完成")
            else:
                logging.warning("⚠️ Embedding 模型檢查失敗，但服務仍會啟動")
                
        except Exception as e:
            logging.warning(f"⚠️ Embedding 模型初始化檢查失敗: {e}")
            logging.info("服務仍會啟動，但可能需要手動處理模型問題")
        
        # 導入 rag_web 來註冊路由
        logging.info("正在導入 rag_web...")
        import rag_web
        logging.info("✅ rag_web 導入成功！")
        
        # 獲取 app
        from rag_web import app
        
        # 列出所有路由
        routes = [f"{route.methods} {route.path}" for route in app.routes if hasattr(route, 'methods')]
        logging.info(f"已註冊的路由: {routes}")
        
        # 啟動服務器
        logging.info("🚀 啟動 RAG Web UI 服務器...")
        logging.info("訪問 http://127.0.0.1:8000/ui 來使用 Web UI")
        logging.info("訪問 http://127.0.0.1:8000/health 來檢查健康狀態")
        
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
        
    except Exception as e:
        logging.error(f"❌ 啟動失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()