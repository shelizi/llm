"""
OpenAI相容API客戶端 - 支援各種OpenAI相容的API服務
"""

import json
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from config_manager import ConfigManager

class OpenAICompatibleClient:
    """OpenAI相容API客戶端"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.session = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """獲取或創建HTTP會話"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=60)  # 60秒超時
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """關閉HTTP會話"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _build_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        """構建對話消息"""
        system_prompt = """你是一個專業的AI助手，專門根據提供的文檔內容回答問題。

請根據以下原則回答：
1. 優先使用提供的文檔內容來回答問題
2. 如果文檔內容不足以完整回答問題，請明確指出
3. 保持回答的準確性和客觀性
4. 使用繁體中文回答
5. 回答要簡潔明瞭，重點突出

文檔內容：
{context}

請基於上述文檔內容回答用戶的問題。"""
        
        return [
            {
                "role": "system",
                "content": system_prompt.format(context=context)
            },
            {
                "role": "user", 
                "content": query
            }
        ]
    
    async def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """使用配置的API生成回答"""
        config = self.config_manager.load_api_config()
        
        if not config or not config.get("enabled", False):
            return {
                "success": False,
                "error": "API未配置或未啟用",
                "answer": ""
            }
        
        try:
            session = await self._get_session()
            
            # 構建請求數據
            messages = self._build_messages(query, context)
            
            request_data = {
                "model": config.get("model_name", "gpt-3.5-turbo"),
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False
            }
            
            # 構建請求標頭
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config['api_token']}"
            }
            
            # 發送請求
            api_url = config["api_url"].rstrip("/")
            if not api_url.endswith("/chat/completions"):
                api_url += "/v1/chat/completions"
            
            logging.info(f"發送請求到: {api_url}")
            
            async with session.post(api_url, json=request_data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # 提取回答
                    if "choices" in result and len(result["choices"]) > 0:
                        answer = result["choices"][0]["message"]["content"]
                        
                        return {
                            "success": True,
                            "answer": answer.strip(),
                            "model": result.get("model", config.get("model_name")),
                            "usage": result.get("usage", {})
                        }
                    else:
                        return {
                            "success": False,
                            "error": "API返回格式錯誤",
                            "answer": ""
                        }
                else:
                    error_text = await response.text()
                    logging.error(f"API請求失敗: {response.status} - {error_text}")
                    
                    return {
                        "success": False,
                        "error": f"API請求失敗: HTTP {response.status}",
                        "answer": ""
                    }
                    
        except asyncio.TimeoutError:
            logging.error("API請求超時")
            return {
                "success": False,
                "error": "API請求超時",
                "answer": ""
            }
        except Exception as e:
            logging.error(f"API請求異常: {e}")
            return {
                "success": False,
                "error": f"API請求異常: {str(e)}",
                "answer": ""
            }
    
    async def test_connection(self, api_url: str, api_token: str, model_name: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """測試API連接"""
        try:
            session = await self._get_session()
            
            # 構建測試請求
            test_messages = [
                {"role": "user", "content": "Hello, this is a connection test."}
            ]
            
            request_data = {
                "model": model_name,
                "messages": test_messages,
                "max_tokens": 10,
                "temperature": 0
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}"
            }
            
            # 處理URL
            test_url = api_url.rstrip("/")
            if not test_url.endswith("/chat/completions"):
                test_url += "/v1/chat/completions"
            
            async with session.post(test_url, json=request_data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "message": "連接測試成功",
                        "model": result.get("model", model_name)
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "message": f"連接測試失敗: HTTP {response.status}",
                        "error": error_text
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "message": f"連接測試失敗: {str(e)}",
                "error": str(e)
            }

# 全局客戶端實例
openai_client = OpenAICompatibleClient()