from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import time
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

# Setul nostru de date demonstrativ (Mini-Dataset)
DOCUMENTS = [
    "Un film despre o mașină rapidă pe străzile din Tokyo.",
    "Un automobil iute de curse câștigă campionatul.",
    "O pisică amuzantă care prinde șoareci în hambar.",
    "O felină adorabilă doarme toată ziua pe canapea.",
    "Programarea în Python este excelentă pentru inteligența artificială.",
    "Baze de date vectoriale sunt folosite pentru căutarea semantică."
]

# Variabile globale pentru a stoca modelele și indexurile în memorie
bm25_index = None
faiss_index = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bm25_index, faiss_index, model
    print("Se încarcă modelul AI și se construiesc indexurile...")
    
    # 1. Inițializare pentru Căutarea Tradițională (BM25)
    tokenized_corpus = [doc.lower().split() for doc in DOCUMENTS]
    bm25_index = BM25Okapi(tokenized_corpus)

    # 2. Inițializare pentru Căutarea Vectorială (FAISS)
    model = SentenceTransformer('all-MiniLM-L6-v2') # Un model rapid și eficient
    document_embeddings = model.encode(DOCUMENTS)
    
    dimension = document_embeddings.shape[1] # Câte dimensiuni are vectorul (ex: 384)
    faiss_index = faiss.IndexFlatL2(dimension) # Folosim Distanța Euclidiană
    faiss_index.add(document_embeddings)
    
    print("Sistemul este gata!")
    yield
    # Aici ar veni codul de curățare la oprirea serverului, dacă ar fi necesar

app = FastAPI(lifespan=lifespan)

# Modelul de date pe care îl primim de la Frontend
class SearchRequest(BaseModel):
    query: str

@app.get("/")
def serve_frontend():
    # Trimitem interfața HTML când utilizatorul accesează browserul
    return FileResponse("app/index.html")

@app.post("/api/search")
def perform_search(request: SearchRequest):
    q = request.query
    
    # --- Traseul 1: Căutare Tradițională (BM25) ---
    start_time_bm25 = time.time()
    tokenized_query = q.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    # Luăm primele 2 cele mai bune rezultate
    top_n_bm25 = np.argsort(bm25_scores)[::-1][:2] 
    # Filtrăm rezultatele cu scor 0 (niciun cuvânt nu s-a potrivit)
    bm25_results = [DOCUMENTS[i] for i in top_n_bm25 if bm25_scores[i] > 0]
    time_bm25 = (time.time() - start_time_bm25) * 1000 # Convertim în milisecunde

    # --- Traseul 2: Căutare Vectorială (FAISS) ---
    start_time_vector = time.time()
    query_vector = model.encode([q])
    # Căutăm cei mai apropiați 2 vecini
    distances, indices = faiss_index.search(query_vector, 2) 
    vector_results = [DOCUMENTS[i] for i in indices[0]]
    time_vector = (time.time() - start_time_vector) * 1000

    return {
        "traditional": {
            "results": bm25_results if bm25_results else ["Niciun rezultat exact găsit."],
            "time_ms": round(time_bm25, 2)
        },
        "vectorial": {
            "results": vector_results,
            "time_ms": round(time_vector, 2)
        }
    }