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
document_categories = []  # Nou: stocăm categoriile documentelor

def highlight_keywords(text, keywords):
    """Evidențiază cuvintele cheie în text folosind HTML tags"""
    import re
    highlighted_text = text
    
    for keyword in keywords:
        # Folosim regex pentru a găsi cuvântul exact (case-insensitive)
        pattern = re.compile(r'(?i)\b' + re.escape(keyword) + r'\b')
        highlighted_text = pattern.sub(f'<mark>{keyword}</mark>', highlighted_text)
    
    return highlighted_text

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bm25_index, faiss_index, model, documents, document_categories
    
    print("1. Descărcăm setul de date '20 Newsgroups'...")
    # Preia texte reale (fără headere de email pentru a păstra textul curat)
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    # Filtrăm textele goale sau prea scurte și luăm primele 2000 de documente
    raw_docs = [doc.replace('\n', ' ').strip() for doc in newsgroups.data if len(doc.strip()) > 50]
    documents = raw_docs[:2000]
    
    # Extragem categoriile corespunzătoare documentelor selectate
    valid_indices = [i for i, doc in enumerate(newsgroups.data) if len(doc.strip()) > 50][:2000]
    document_categories = [newsgroups.target_names[newsgroups.target[i]] for i in valid_indices]

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

class DocumentRequest(BaseModel):
    id: int
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
    bm25_results = []
    for i in top_n_bm25:
        if bm25_scores[i] > 0:
            category = document_categories[i]
            text_snippet = documents[i][:300] + "..."
            highlighted_snippet = highlight_keywords(text_snippet, tokenized_query)
            highlighted_full_text = highlight_keywords(documents[i], tokenized_query)
            bm25_results.append({
                "id": int(i),
                "category": category,
                "snippet": highlighted_snippet,
                "full_text": highlighted_full_text
            })
    time_bm25 = (time.time() - start_time_bm25) * 1000

    # --- Traseul 2: Căutare Vectorială (FAISS) ---
    start_time_vector = time.time()
    query_vector = model.encode([q])
    
    # Căutăm top 3 rezultate
    distances, indices = faiss_index.search(query_vector, 3) 
    
    vector_results = []
    for dist, idx in zip(distances[0], indices[0]):
        category = document_categories[idx]
        text_snippet = documents[idx][:300] + "..."
        vector_results.append({
            "id": int(idx),
            "category": category,
            "snippet": text_snippet,
            "full_text": documents[idx],
            "distance": float(round(dist, 2))
        })
        
    time_vector = (time.time() - start_time_vector) * 1000

    return {
        "traditional": {
            "results": bm25_results,
            "time_ms": round(time_bm25, 2)
        },
        "vectorial": {
            "results": vector_results,
            "time_ms": round(time_vector, 2)
        }
    }

@app.post("/api/document")
def get_document(request: DocumentRequest):
    doc_id = request.id
    query = request.query
    
    if doc_id < 0 or doc_id >= len(documents):
        return {"error": "Document not found"}
    
    # Evidențiem cuvintele cheie în documentul complet
    tokenized_query = query.lower().split()
    highlighted_full_text = highlight_keywords(documents[doc_id], tokenized_query)
    
    return {
        "id": doc_id,
        "category": document_categories[doc_id],
        "full_text": highlighted_full_text
    }