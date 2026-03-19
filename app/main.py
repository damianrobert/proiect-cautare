import time
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sklearn.datasets import fetch_20newsgroups

# Variabile globale
bm25_index = None
faiss_index = None
model = None
documents = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bm25_index, faiss_index, model, documents
    
    print("1. Descărcăm setul de date '20 Newsgroups'...")
    # Preia texte reale (fără headere de email pentru a păstra textul curat)
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    # Filtrăm textele goale sau prea scurte și luăm primele 2000 de documente
    raw_docs = [doc.replace('\n', ' ').strip() for doc in newsgroups.data if len(doc.strip()) > 50]
    documents = raw_docs[:2000]

    print(f"2. Am încărcat {len(documents)} documente. Construim indexul Tradițional (BM25)...")
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25_index = BM25Okapi(tokenized_corpus)

    print("3. Încărcăm modelul AI (MiniLM)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("4. Transformăm documentele în vectori (Acest proces va dura 10-30 de secunde)...")
    document_embeddings = model.encode(documents)
    
    dimension = document_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(document_embeddings)
    
    print("Sistemul este complet gata de utilizare!")
    yield

app = FastAPI(lifespan=lifespan)

class SearchRequest(BaseModel):
    query: str

@app.get("/")
def serve_frontend():
    return FileResponse("app/index.html")

@app.post("/api/search")
def perform_search(request: SearchRequest):
    q = request.query
    
    # --- Traseul 1: Căutare Tradițională (BM25) ---
    start_time_bm25 = time.time()
    tokenized_query = q.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    # Luăm top 3 rezultate care au scor > 0
    top_n_bm25 = np.argsort(bm25_scores)[::-1][:3]
    bm25_results = [documents[i][:300] + "..." for i in top_n_bm25 if bm25_scores[i] > 0]
    time_bm25 = (time.time() - start_time_bm25) * 1000

    # --- Traseul 2: Căutare Vectorială (FAISS) ---
    start_time_vector = time.time()
    query_vector = model.encode([q])
    
    # Căutăm top 3 rezultate
    distances, indices = faiss_index.search(query_vector, 3) 
    
    vector_results = []
    for dist, idx in zip(distances[0], indices[0]):
        # Formatăm rezultatul pentru a arăta și distanța matematică
        score_text = f"<strong>[Distanță L2: {round(dist, 2)}]</strong><br>"
        text_snippet = documents[idx][:300] + "..."
        vector_results.append(score_text + text_snippet)
        
    time_vector = (time.time() - start_time_vector) * 1000

    return {
        "traditional": {
            "results": bm25_results if bm25_results else ["Niciun document nu conține aceste cuvinte exacte."],
            "time_ms": round(time_bm25, 2)
        },
        "vectorial": {
            "results": vector_results,
            "time_ms": round(time_vector, 2)
        }
    }