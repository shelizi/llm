"""Web UI for RAG file 管理與索引建立。
此模組在匯入後會把路由掛載到現有的 `rag_api.app`。
"""

from __future__ import annotations

from pathlib import Path
import logging
import shutil
from typing import Dict

import chromadb
import json
import docx
import PyPDF2
from fastapi import BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from llama_index import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.text_splitter import TokenTextSplitter
from llama_index import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

import torch

# 取用既有的 FastAPI instance 與設定
import rag_api  # noqa
from rag_api import app, CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME, DEVICE, index

# ----------------- 路徑與模板 -----------------
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ----------------- 狀態管理 -----------------
STATUS_FILE = BASE_DIR / "status.json"

# 讀取現有狀態，若無則建立空 dict
if STATUS_FILE.exists():
    try:
        StatusDict: Dict[str, str] = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        StatusDict: Dict[str, str] = {}
else:
    StatusDict: Dict[str, str] = {}

def save_status() -> None:
    """將 StatusDict 寫入磁碟"""
    try:
        STATUS_FILE.write_text(json.dumps(StatusDict, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logging.error("無法寫入狀態檔: %s", e)

# ----------------- 輔助函式 -----------------

def read_file_text(filepath: Path) -> str:
    suffix = filepath.suffix.lower()
    if suffix == ".txt":
        return filepath.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        reader = PyPDF2.PdfReader(str(filepath))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in (".docx", ".doc"):
        document = docx.Document(str(filepath))
        return "\n".join(p.text for p in document.paragraphs)
    else:
        raise ValueError("Unsupported file type: " + suffix)


def process_and_index_file(filename: str, filepath: Path, chunk_size: int, overlap: int):
    """背景任務：處理檔案並寫入向量資料庫"""
    try:
        StatusDict[filename] = "索引中"
        save_status()
        text = read_file_text(filepath)
        documents = [Document(text=text, metadata={"filename": filename})]

        # 準備向量存取
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        vector_store = ChromaVectorStore(chroma_client=client, collection_name=COLLECTION_NAME)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device=DEVICE)
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

        global index  # 取用 rag_api 的 index 變數
        if index is None:
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embed_model,
                text_splitter=text_splitter,
            )
        else:
            # 已有索引 -> 直接插入文件
            index.insert_documents(documents, text_splitter=text_splitter)

        StatusDict[filename] = "已索引"
        save_status()
        logging.info("✅ %s 已完成索引", filename)
    except Exception as e:
        StatusDict[filename] = f"錯誤: {e}"
        save_status()
        logging.error("❌ 索引檔案 %s 失敗: %s", filename, e)


# ----------------- 路由 -----------------

@app.get("/ui", response_class=HTMLResponse)
async def ui_root(request: Request):
    files = sorted([p.name for p in UPLOAD_DIR.iterdir() if p.is_file()])
    statuses = {f: StatusDict.get(f, "未索引") for f in files}
    return templates.TemplateResponse("index.html", {"request": request, "files": files, "statuses": statuses})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    StatusDict[file.filename] = "已上傳"
    save_status()
    return RedirectResponse("/ui", status_code=303)


@app.post("/index")
async def index_file(background_tasks: BackgroundTasks, filename: str = Form(...), chunk_size: int = Form(512), overlap: int = Form(50)):
    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    background_tasks.add_task(process_and_index_file, filename, filepath, chunk_size, overlap)
    return RedirectResponse("/ui", status_code=303)


@app.get("/status")
async def status():
    return StatusDict
