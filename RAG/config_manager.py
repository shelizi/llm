"""
配置管理模組 - 處理API設定的加密存儲和讀取
"""

import json
import os
from pathlib import Path
from cryptography.fernet import Fernet
from typing import Dict, Optional
import logging

class ConfigManager:
    """管理加密的配置設定"""
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path(__file__).parent / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        self.key_file = self.config_dir / "config.key"
        self.config_file = self.config_dir / "api_config.enc"
        
        # 初始化或載入加密金鑰
        self.cipher_suite = self._init_cipher()
        
    def _init_cipher(self) -> Fernet:
        """初始化或載入加密金鑰"""
        if self.key_file.exists():
            # 載入現有金鑰
            key = self.key_file.read_bytes()
        else:
            # 生成新金鑰
            key = Fernet.generate_key()
            self.key_file.write_bytes(key)
            # 設定文件權限（僅所有者可讀寫）
            os.chmod(self.key_file, 0o600)
            
        return Fernet(key)
    
    def save_api_config(self, api_url: str, api_token: str, model_name: str = "gpt-3.5-turbo") -> bool:
        """保存API配置（加密）"""
        try:
            config_data = {
                "api_url": api_url.strip(),
                "api_token": api_token.strip(),
                "model_name": model_name.strip(),
                "enabled": True
            }
            
            # 序列化並加密
            json_data = json.dumps(config_data, ensure_ascii=False)
            encrypted_data = self.cipher_suite.encrypt(json_data.encode('utf-8'))
            
            # 寫入文件
            self.config_file.write_bytes(encrypted_data)
            
            # 設定文件權限
            os.chmod(self.config_file, 0o600)
            
            logging.info("API配置已成功保存")
            return True
            
        except Exception as e:
            logging.error(f"保存API配置失敗: {e}")
            return False
    
    def load_api_config(self) -> Optional[Dict]:
        """載入API配置（解密）"""
        try:
            if not self.config_file.exists():
                return None
                
            # 讀取並解密
            encrypted_data = self.config_file.read_bytes()
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            # 解析JSON
            config_data = json.loads(decrypted_data.decode('utf-8'))
            
            # 驗證必要欄位
            required_fields = ["api_url", "api_token", "model_name"]
            if all(field in config_data for field in required_fields):
                return config_data
            else:
                logging.warning("配置文件缺少必要欄位")
                return None
                
        except Exception as e:
            logging.error(f"載入API配置失敗: {e}")
            return None
    
    def delete_api_config(self) -> bool:
        """刪除API配置"""
        try:
            if self.config_file.exists():
                self.config_file.unlink()
                logging.info("API配置已刪除")
            return True
        except Exception as e:
            logging.error(f"刪除API配置失敗: {e}")
            return False
    
    def is_api_configured(self) -> bool:
        """檢查是否已配置API"""
        config = self.load_api_config()
        return config is not None and config.get("enabled", False)
    
    def get_api_status(self) -> Dict:
        """獲取API配置狀態（不包含敏感信息）"""
        config = self.load_api_config()
        if config:
            return {
                "configured": True,
                "api_url": config.get("api_url", ""),
                "model_name": config.get("model_name", ""),
                "enabled": config.get("enabled", False),
                "token_length": len(config.get("api_token", ""))
            }
        else:
            return {
                "configured": False,
                "api_url": "",
                "model_name": "",
                "enabled": False,
                "token_length": 0
            }