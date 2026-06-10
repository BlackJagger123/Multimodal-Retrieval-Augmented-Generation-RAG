import os
import time
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from PIL import Image
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =================================================================
# 1. INISIALISASI GLOBAL (Menghindari Cold Start)
# =================================================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di environment variables!")

# Menginisiasi Gemini dan Model Embedding Lokal
client = genai.Client(api_key=api_key)
model_embedding_lokal = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Menginisiasi Database Vektor
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

# Schema penerima untuk endpoint query
class RequestPertanyaan(BaseModel):
    pertanyaan: str

# =================================================================
# 3. ENDPOINT 1: INGESTION (Upload & Process PDF)
# =================================================================
@app.post("/ingest")
async def ingest_dokumen(file: UploadFile = File(...)):
    """
    Endpoint ini akan menerima file PDF, melakukan pemotongan (Chunking), 
    menggunakan VLM untuk ekstraksi, dan menyimpannya ke Vector Database (ChromaDB).
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berupa PDF!")
    
    # Simpan PDF sementara
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        print("Mulai konversi PDF ke gambar...")
        # Tambahkan r"..." di depan path agar Windows tidak bingung membaca tanda backslash (\)
        jalur_poppler = r"C:\poppler\Library\bin"
        halaman_gambar = convert_from_path(temp_pdf_path, dpi=200, poppler_path=jalur_poppler)
        
        dokumen_langchain = []
        
        prompt_ekstraksi = """
        Ekstrak seluruh teks dari gambar halaman ini.
        JIKA terdapat tabel, ubah tabel tersebut menjadi format Markdown tabel yang sangat rapi.
        JIKA terdapat grafik/infografis, deskripsikan datanya secara mendetail.
        JANGAN menambahkan narasi pembuka/penutup, cukup berikan hasil ekstraksinya saja.
        """
        
        print("Mulai ekstraksi VLM (Harap bersabar, proses ini memakan waktu)...")
        for i, gambar in enumerate(halaman_gambar):
            print(f"Memproses Halaman {i+1}/{len(halaman_gambar)}...")
            
            # Simpan halaman sebagai gambar sementara
            temp_img_path = f"temp_page_{i+1}.png"
            gambar.save(temp_img_path, "PNG")
            img_pil = Image.open(temp_img_path)
            
            # Tembak ke Gemini 2.5 Flash
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=[img_pil, prompt_ekstraksi]
            )
            
            teks_hasil = response.text
            
            # Gabungkan teks dengan Metadata Halaman
            doc = Document(
                page_content=teks_hasil,
                metadata={"page": i + 1, "source": file.filename}
            )
            dokumen_langchain.append(doc)
            
            # Bersihkan cache gambar halaman dan beri jeda waktu API
            img_pil.close()
            os.remove(temp_img_path)
            time.sleep(5) 
            
        print("Melakukan Chunking teks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, 
            chunk_overlap=500,
            separators=["\n\n", "\n", " ", ""] 
        )
        potongan_dokumen = text_splitter.split_documents(dokumen_langchain)
        
        print("Menyimpan ke Vector Database...")
        vector_db.add_documents(potongan_dokumen)
        
        return {
            "status": "success", 
            "pesan": f"File {file.filename} berhasil diproses!",
            "total_halaman": len(halaman_gambar),
            "total_chunks": len(potongan_dokumen)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat pemrosesan: {str(e)}")
        
    finally:
        # Selalu bersihkan file PDF asli agar memori tidak penuh
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

# =================================================================
# 4. ENDPOINT 2: QUERY (Menjawab Pertanyaan)
# =================================================================
@app.post("/query")
async def tanya_dokumen(payload: RequestPertanyaan):
    """
    Endpoint ini menerima pertanyaan dari user, mencari konteks di ChromaDB,
    lalu menggunakan Gemini 3.5 Flash untuk menjawabnya beserta sitasi halaman.
    """
    pertanyaan_user = payload.pertanyaan
    
    # A. Retrieval (Pencarian dokumen di ChromaDB)
    hasil_pencarian = vector_db.similarity_search(pertanyaan_user, k=15)
    konteks = ""
    for doc in hasil_pencarian:
        halaman = doc.metadata.get('page', 'Tidak diketahui')
        konteks += f"---\n [SUMBER: HALAMAN {halaman}]\n{doc.page_content} \n"
        
    # B. Generation (Prompting super ketat ke Gemini)
    prompt_rag = f"""
    Kamu adalah asisten AI yang cerdas dan ahli dalam menganalisis dokumen keuangan Bank Mandiri.
    Tugasmu adalah menjawab pertanyaan menggunakan HANYA informasi dari 'Konteks Dokumen' di bawah ini.
    Setiap potongan konteks diawali dengan tag [SUMBER: HALAMAN X].
    
    Aturan Wajib:
    1. Sajikan data angka nominal dan persentase dengan sangat akurat.
    2. JABARKAN RINCIAN: Jika konteks berisi informasi dalam bentuk daftar, poin-poin, atau data infografis (seperti daftar saluran, tahapan, dll), kamu WAJIB menyebutkan seluruh poinnya satu per satu secara spesifik dan detail. Jangan diringkas atau dilewati.
    3. Di akhir jawabanmu, buatlah satu baris khusus berbunyi: "Sumber Halaman: [Sebutkan nomor halamannya]". 
    4. SITASI SUPER KETAT: Kamu HANYA boleh mengutip halaman yang benar-benar memuat tabel, grafik, atau angka eksak tersebut secara fisik. JANGAN mengutip halaman yang hanya berisi narasi pengantar atau kata kunci yang mirip. Jika angkanya murni ditarik dari Halaman 4, HANYA sebutkan Halaman 4.
    
    Konteks Dokumen:
    {konteks}

    Pertanyaan: {pertanyaan_user}
    Jawaban:
    """

    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt_rag
        )
        return {
            "pertanyaan": pertanyaan_user,
            "jawaban": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))