import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =================================================================
# 1. INISIALISASI GLOBAL (Menghindari Cold Start)
# =================================================================
# Kita memuat DB dan Model di luar fungsi endpoint agar sistem 
# tidak perlu men-download/memuat ulang model setiap kali ada user yang bertanya.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di environment variables!")

client = genai.Client(api_key=api_key)
model_embedding_lokal = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
direktori_db = "./mandiri_chroma_db"
vector_db = Chroma(persist_directory=direktori_db, embedding_function=model_embedding_lokal)

# =================================================================
# 2. SETUP FASTAPI & PYDANTIC SCHEMA
# =================================================================
app = FastAPI(
    title="Mandiri Multimodal RAG API",
    description="API untuk memproses dan tanya-jawab Laporan Keuangan Bank Mandiri",
    version="1.0.0"
)

# Pydantic Schema: Cetakan data untuk menerima pertanyaan
class RequestPertanyaan(BaseModel):
    pertanyaan: str

# =================================================================
# 3. ENDPOINT 1: INGESTION (Upload & Process PDF)
# =================================================================
@app.post("/ingest")
async def ingest_dokumen(file: UploadFile = File(...)):
    """
    Endpoint ini akan menerima file PDF, melakukan pemotongan (Chunking), 
    dan menyimpannya ke Vector Database (ChromaDB).
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berupa PDF!")
    
    # [AREA TUGAS SELANJUTNYA] 
    # Di sinilah nanti kita akan memindahkan logika PDF -> JSON -> ChromaDB
    
    return {"status": "success", "pesan": f"File {file.filename} diterima. (Logika pemrosesan akan ditambahkan)"}

# =================================================================
# 4. ENDPOINT 2: QUERY (Menjawab Pertanyaan)
# =================================================================
@app.post("/query")
async def tanya_dokumen(payload: RequestPertanyaan):
    """
    Endpoint ini menerima pertanyaan dari user, mencari konteks di ChromaDB,
    lalu menggunakan Gemini 2.5 Flash untuk menjawabnya.
    """
    pertanyaan_user = payload.pertanyaan
    
    # A. Retrieval (Pencarian dokumen)
    hasil_pencarian = vector_db.similarity_search(pertanyaan_user, k=15)
    konteks = ""
    for doc in hasil_pencarian:
        halaman = doc.metadata.get('page', 'Tidak diketahui')
        konteks += f"---\n[SUMBER: HALAMAN {halaman}]\n{doc.page_content}\n"
        
    # B. Generation (Prompting ke Gemini 2.5 Flash)
    prompt_rag = f"""
    Kamu adalah asisten AI yang cerdas dan ahli dalam menganalisis dokumen keuangan Bank Mandiri.
    Tugasmu adalah menjawab pertanyaan menggunakan HANYA informasi dari 'Konteks Dokumen' di bawah ini.
    Setiap potongan konteks diawali dengan tag [SUMBER: HALAMAN X].
    
    Aturan Wajib:
    1. Sajikan data angka nominal dan persentase dengan sangat akurat.
    2. JABARKAN RINCIAN: Jika konteks berisi informasi dalam bentuk daftar, poin-poin, atau data infografis (seperti daftar saluran, tahapan, dll), kamu WAJIB menyebutkan seluruh poinnya satu per satu secara spesifik dan detail. Jangan diringkas atau dilewati.
    3. Di akhir jawabanmu, buatlah satu baris khusus berbunyi: "📌 Sumber Halaman: [Sebutkan nomor halamannya]". 
    4. HANYA sebutkan halaman tempat kamu benar-benar menemukan angka/jawaban tersebut. Jangan sebutkan halaman yang tidak relevan.

    Konteks Dokumen:
    {konteks}

    Pertanyaan: {pertanyaan_user}
    Jawaban:
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Menggunakan 2.5 Flash sesuai kesepakatan
            contents=prompt_rag
        )
        return {
            "pertanyaan": pertanyaan_user,
            "jawaban": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))