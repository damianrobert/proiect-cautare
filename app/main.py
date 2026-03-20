import time
import numpy as np
import faiss
import pickle
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sklearn.datasets import fetch_20newsgroups

# Variabile globale
bm25_index = None
faiss_index = None
model = None
cross_encoder = None
documents = []
document_categories = []  # Nou: stocăm categoriile documentelor

# Fișiere pentru persistență
FAISS_INDEX_FILE = "faiss_index.bin"
DOCUMENTS_FILE = "documents.pkl"
BM25_FILE = "bm25_index.pkl"

def highlight_keywords(text, keywords):
    """Evidențiază cuvintele cheie în text folosind HTML tags"""
    import re
    highlighted_text = text
    
    for keyword in keywords:
        # Folosim regex pentru a găsi cuvântul exact (case-insensitive)
        pattern = re.compile(r'(?i)\b' + re.escape(keyword) + r'\b')
        highlighted_text = pattern.sub(f'<mark>{keyword}</mark>', highlighted_text)
    
    return highlighted_text

def save_to_disk():
    """Salvează indexele și documentele pe disc"""
    print("5. Salvăm datele pe disc pentru cache...")
    
    # Salvăm indexul FAISS
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    
    # Salvăm documentele și categoriile
    with open(DOCUMENTS_FILE, 'wb') as f:
        pickle.dump({
            'documents': documents,
            'categories': document_categories
        }, f)
    
    # Salvăm indexul BM25
    with open(BM25_FILE, 'wb') as f:
        pickle.dump(bm25_index, f)
    
    print("Datele au fost salvate cu succes!")

def load_from_disk():
    """Încarcă indexele și documentele de pe disc"""
    if not all(os.path.exists(f) for f in [FAISS_INDEX_FILE, DOCUMENTS_FILE, BM25_FILE]):
        return False
    
    print("Încărcăm datele din cache...")
    
    # Încărcăm indexul FAISS
    faiss_index_loaded = faiss.read_index(FAISS_INDEX_FILE)
    
    # Încărcăm documentele și categoriile
    with open(DOCUMENTS_FILE, 'rb') as f:
        data = pickle.load(f)
        documents_loaded = data['documents']
        categories_loaded = data['categories']
    
    # Încărcăm indexul BM25
    with open(BM25_FILE, 'rb') as f:
        bm25_index_loaded = pickle.load(f)
    
    return faiss_index_loaded, documents_loaded, categories_loaded, bm25_index_loaded

def cross_encoder_rerank(query, candidate_docs, top_k=5):
    """
    Re-ranking cu Cross-Encoder - arhitectură în 2 etape
    Etapa 1: Bi-Encoder/BM25 pentru filtrare rapidă (candidați)
    Etapa 2: Cross-Encoder pentru re-ranking fin (acuratețe)
    """
    if not candidate_docs:
        return []
    
    # Pregătim perechile (query, document) pentru Cross-Encoder
    cross_inputs = [[query, doc['full_text']] for doc in candidate_docs]
    
    # Cross-Encoder scoruri - citește query și document simultan
    cross_scores = cross_encoder.predict(cross_inputs)
    
    # Adăugăm scorurile Cross-Encoder la documente
    for i, doc in enumerate(candidate_docs):
        doc['cross_score'] = float(round(cross_scores[i], 4))
    
    # Sortăm după scorul Cross-Encoder descrescător
    reranked_docs = sorted(candidate_docs, key=lambda x: x['cross_score'], reverse=True)
    
    return reranked_docs[:top_k]

def hybrid_search_with_reranking(query, top_candidates=20, final_results=5):
    """
    Pipeline complet: BM25 + Vector → RRF → Cross-Encoder Re-ranking
    """
    # Etapa 1: Căutare BM25 (extinsă pentru mai mulți candidați)
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_n_bm25 = np.argsort(bm25_scores)[::-1][:top_candidates]
    
    bm25_candidates = []
    for i in top_n_bm25:
        if bm25_scores[i] > 0:
            bm25_candidates.append({
                "id": int(i),
                "category": document_categories[i],
                "snippet": documents[i][:300] + "...",
                "full_text": documents[i],
                "bm25_score": float(round(bm25_scores[i], 2))
            })
    
    # Etapa 2: Căutare Vectorială (extinsă)
    query_vector = model.encode([query])
    distances, indices = faiss_index.search(query_vector, top_candidates)
    
    vector_candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        similarity = float(round(np.exp(-dist) * 100, 1))
        vector_candidates.append({
            "id": int(idx),
            "category": document_categories[idx],
            "snippet": documents[idx][:300] + "...",
            "full_text": documents[idx],
            "similarity": similarity
        })
    
    # Etapa 3: Fuziune RRF pentru top candidați
    hybrid_candidates = reciprocal_rank_fusion(bm25_candidates, vector_candidates)
    
    # Etapa 4: Cross-Encoder Re-ranking
    final_results = cross_encoder_rerank(query, hybrid_candidates, final_results)
    
    return final_results

def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    """
    Reciprocal Rank Fusion (RRF) - combină rezultatele din multiple sisteme de căutare
    Formula: 1 / (k + rank) unde k este constantă (de obicei 60)
    """
    fusion_scores = {}
    
    # Adăugăm scorurile BM25 (rank-ul pornește de la 1)
    for rank, result in enumerate(bm25_results, 1):
        doc_id = result['id']
        rrf_score = 1.0 / (k + rank)
        fusion_scores[doc_id] = {
            'rrf_score': rrf_score,
            'bm25_rank': rank,
            'vector_rank': None,
            'category': result['category'],
            'snippet': result['snippet'],
            'full_text': result['full_text'],
            'bm25_score': result['bm25_score']
        }
    
    # Adăugăm scorurile Vectoriale
    for rank, result in enumerate(vector_results, 1):
        doc_id = result['id']
        rrf_score = 1.0 / (k + rank)
        
        if doc_id in fusion_scores:
            # Documentul apare în ambele rezultate - combinăm scorurile
            fusion_scores[doc_id]['rrf_score'] += rrf_score
            fusion_scores[doc_id]['vector_rank'] = rank
            fusion_scores[doc_id]['similarity'] = result['similarity']
        else:
            # Documentul apare doar în rezultatele vectoriale
            fusion_scores[doc_id] = {
                'rrf_score': rrf_score,
                'bm25_rank': None,
                'vector_rank': rank,
                'category': result['category'],
                'snippet': result['snippet'],
                'full_text': result['full_text'],
                'similarity': result['similarity']
            }
    
    # Sortăm după scorul RRF descrescător
    sorted_results = sorted(fusion_scores.items(), key=lambda x: x[1]['rrf_score'], reverse=True)
    
    # Returnăm top 5 rezultate hibride
    hybrid_results = []
    for doc_id, data in sorted_results[:5]:
        result = {
            'id': doc_id,
            'category': data['category'],
            'snippet': data['snippet'],
            'full_text': data['full_text'],
            'rrf_score': float(round(data['rrf_score'], 4)),
            'bm25_rank': data['bm25_rank'],
            'vector_rank': data['vector_rank']
        }
        
        # Adăugăm scorurile specifice dacă există
        if 'bm25_score' in data:
            result['bm25_score'] = data['bm25_score']
        if 'similarity' in data:
            result['similarity'] = data['similarity']
            
        hybrid_results.append(result)
    
    return hybrid_results

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bm25_index, faiss_index, model, cross_encoder, documents, document_categories
    
    # Încercăm să încărcăm din cache
    cached_data = load_from_disk()
    if cached_data:
        faiss_index, documents, document_categories, bm25_index = cached_data
        print("Datele au fost încărcate din cache în < 1 secundă!")
    else:
        print("Nu am găsit cache. Construim totul de la zero...")
        
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
        
        # Salvăm pentru viitoarele rulări
        save_to_disk()
    
    # Asigurăm că modelul este încărcat (nu îl salvăm în cache)
    if model is None:
        print("Încărcăm modelul AI (MiniLM)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Încărcăm Cross-Encoder pentru re-ranking
    if cross_encoder is None:
        print("Încărcăm Cross-Encoder pentru re-ranking fin...")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
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
                "full_text": highlighted_full_text,
                "bm25_score": float(round(bm25_scores[i], 2))
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
        
        # Transformăm distanța L2 în procentaj de similaritate
        # Folosim formula: similarity = exp(-distance) * 100
        # Astfel, distanță 0 -> 100%, distanță mare -> aproape 0%
        similarity = float(round(np.exp(-dist) * 100, 1))
        
        vector_results.append({
            "id": int(idx),
            "category": category,
            "snippet": text_snippet,
            "full_text": documents[idx],
            "similarity": similarity
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

@app.post("/api/hybrid-search")
def perform_hybrid_search(request: SearchRequest):
    q = request.query
    
    # --- Traseul 1: Căutare Tradițională (BM25) ---
    start_time_bm25 = time.time()
    tokenized_query = q.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    # Luăm top 5 rezultate pentru RRF
    top_n_bm25 = np.argsort(bm25_scores)[::-1][:5]
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
                "full_text": highlighted_full_text,
                "bm25_score": float(round(bm25_scores[i], 2))
            })
    time_bm25 = (time.time() - start_time_bm25) * 1000

    # --- Traseul 2: Căutare Vectorială (FAISS) ---
    start_time_vector = time.time()
    query_vector = model.encode([q])
    
    # Căutăm top 5 rezultate pentru RRF
    distances, indices = faiss_index.search(query_vector, 5) 
    
    vector_results = []
    for dist, idx in zip(distances[0], indices[0]):
        category = document_categories[idx]
        text_snippet = documents[idx][:300] + "..."
        
        # Transformăm distanța L2 în procentaj de similaritate
        similarity = float(round(np.exp(-dist) * 100, 1))
        
        vector_results.append({
            "id": int(idx),
            "category": category,
            "snippet": text_snippet,
            "full_text": documents[idx],
            "similarity": similarity
        })
        
    time_vector = (time.time() - start_time_vector) * 1000

    # --- Traseul 3: Fuziune Hibridă (RRF) ---
    start_time_hybrid = time.time()
    hybrid_results = reciprocal_rank_fusion(bm25_results, vector_results)
    time_hybrid = (time.time() - start_time_hybrid) * 1000

    return {
        "hybrid": {
            "results": hybrid_results,
            "time_ms": round(time_hybrid, 2),
            "total_time_ms": round(time_bm25 + time_vector + time_hybrid, 2)
        }
    }

@app.post("/api/rerank-search")
def perform_rerank_search(request: SearchRequest):
    q = request.query
    
    # Pipeline complet: BM25 + Vector → RRF → Cross-Encoder
    start_time_total = time.time()
    
    # Etapa 1-4: Pipeline complet cu Cross-Encoder re-ranking
    final_results = hybrid_search_with_reranking(q, top_candidates=20, final_results=5)
    
    # Evidențiem cuvintele cheie în rezultatele finale
    tokenized_query = q.lower().split()
    for result in final_results:
        result['snippet'] = highlight_keywords(result['snippet'], tokenized_query)
    
    total_time = (time.time() - start_time_total) * 1000
    
    return {
        "rerank": {
            "results": final_results,
            "time_ms": round(total_time, 2),
            "pipeline": "BM25 + Vector → RRF → Cross-Encoder"
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