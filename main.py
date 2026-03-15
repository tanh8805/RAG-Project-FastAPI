import httpx
from docx import Document as DocxDocument
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import io
import psycopg2
from dotenv import load_dotenv
import os
import requests

load_dotenv()

app = FastAPI()

print('Đang tải Embedding Model...')
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print('Tải Embedding Model Thành Công')



@app.post('/api/process-document')
async def process_document(
    document_id : str = Form(),
    file_url: str = Form(),    
    file_name: str = Form()
):
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(file_url)
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="Không thể download file từ MinIO")
        file_bytes = response.content

    text_content = ""
    if file_name.endswith(".pdf"):
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        for page in pdf_reader.pages:
            text_content += (page.extract_text() or "") + "\n"
    elif file_name.endswith(".docx") or file_name.endswith(".doc"):
        docx_file = DocxDocument(io.BytesIO(file_bytes))
        for para in docx_file.paragraphs:
            text_content += para.text + "\n"
    else:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ .pdf, .doc, .docx")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
    chunks = text_splitter.split_text(text_content)

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )
    cursor = conn.cursor()

    try:
        for i,chunk in enumerate(chunks):
            vector = embedding_model.encode(chunk).tolist()
            cursor.execute(
                """
                INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """, (document_id, i, chunk, vector)
            )
        conn.commit()
    except Exception as e:
        conn.rollback() 
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu Database: {str(e)}")
    finally:
        cursor.close()
        conn.close()
    return {
        "status": "success",
        "message": f"Đã xử lý và lưu thành công {len(chunks)} đoạn văn bản."
    }

@app.post('/api/ask-AI')
async def ask_AI(
    document_id : str = Form(),
    query : str = Form()
):
    try:
        query_vector = embedding_model.encode(query).tolist()

        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT")
        )
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT content, chunk_index
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY embedding <=> %s
            LIMIT 5
            """, (document_id, query_vector)
        )
        result = cursor.fetchall()

        if not result:
            return {
                "status": "success",
                "answer": "Không tìm thấy thông tin trong tài liệu."
            }

        context = "\n\n".join([r[0] for r in result])

        prompt = f"""
                Bạn là trợ lý AI trả lời dựa trên tài liệu.

                Chỉ sử dụng thông tin trong context.

                Context:
                {context}

                Question:
                {query}

                Nếu context không chứa câu trả lời thì hãy nói:
                "Không tìm thấy trong tài liệu".
                """
        response = requests.post(
            "http://ollama:11434/api/generate",
            json={
                "model" : "llama3",
                "prompt": prompt,
                "stream" : False
            }
        )

        answer = response.json()['response']
        return {
            "status" : "success",
            "answer" : answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")
    finally:
        cursor.close()
        conn.close()