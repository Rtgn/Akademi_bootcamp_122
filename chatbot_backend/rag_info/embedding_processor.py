import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import os

class EmbeddingProcessor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
       
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.vector_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model yüklendi: {model_name}")
        print(f"Vektör boyutu: {self.vector_dimension}")
        
    def load_chunks(self, file_path: str = "processed_chunks.jsonl") -> List[Dict[str, Any]]:
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line.strip()))
        print(f"{len(chunks)} chunk yüklendi")
        return chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> tuple:
        texts = [chunk["content"] for chunk in chunks]
        
        print("Embedding'ler oluşturuluyor...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        print(f"Embedding'ler oluşturuldu: {embeddings.shape}")
        return embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray, index_type: str = "IVFFlat") -> faiss.Index:
        """FAISS vektör indeksi oluşturur"""
        print(f"FAISS indeksi oluşturuluyor (tip: {index_type})...")
        
        if index_type == "IVFFlat":
           
            nlist = min(100, len(embeddings) // 10)  # Cluster sayısı
            quantizer = faiss.IndexFlatIP(self.vector_dimension)
            index = faiss.IndexIVFFlat(quantizer, self.vector_dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Eğitim verisi ile indeksi eğit
            if len(embeddings) > nlist:
                index.train(embeddings)
            
        elif index_type == "Flat":
            # Basit flat indeks (tam arama)
            index = faiss.IndexFlatIP(self.vector_dimension)
            
        else:
            raise ValueError(f"Desteklenmeyen indeks tipi: {index_type}")
        
        # Embedding'leri indekse ekle
        index.add(embeddings.astype('float32'))
        print(f"İndeks oluşturuldu: {index.ntotal} vektör eklendi")
        
        return index
    
    def save_vector_database(self, index: faiss.Index, chunks: List[Dict[str, Any]], 
                           output_dir: str = "vector_database"):
        """Vektör veri tabanını kaydeder"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # FAISS indeksini kaydesder
        faiss.write_index(index, str(output_path / "faiss_index.bin"))
        
        with open(output_path / "chunks_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        #Model bilgilerini kaydet
        model_info = {
            "model_name": self.model_name,
            "vector_dimension": self.vector_dimension,
            "total_vectors": len(chunks),
            "index_type": "IVFFlat"
        }
        
        with open(output_path / "model_info.json", 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print(f"Vektör veri tabanı kaydedildi: {output_path}")
    
    def test_similarity_search(self, index: faiss.Index, chunks: List[Dict[str, Any]], 
                             query: str = "hamilelik belirtileri", top_k: int = 5):
        print(f"\nTest araması: '{query}'")
        
        # Sorgu embedding'i oluştur
        query_embedding = self.model.encode([query])
        
        # En yakın vektörleri bul
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
        
        print(f"En yakın {top_k} sonuç:")
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            chunk = chunks[idx]
            print(f"\n{i+1}. Benzerlik: {distance:.4f}")
            print(f"   Dosya: {chunk['metadata']['source_file']}")
            print(f"   İçerik: {chunk['content'][:200]}...")

def main():
    processor = EmbeddingProcessor()
    
    chunks = processor.load_chunks()
    
    if not chunks:
        print("Chunk bulunamadı!")
        return
    
    # Embedding'leri oluştur
    embeddings = processor.create_embeddings(chunks)
    
    # FAISS indeksi oluştur
    index = processor.create_faiss_index(embeddings)
    
    # Vektör veri tabanını kaydet
    processor.save_vector_database(index, chunks)
    
    # Test araması yap
    processor.test_similarity_search(index, chunks)
    
    print("\nEmbedding işlemi tamamlandı!")
    print("Vektör veri tabanı 'vector_database' klasöründe oluşturuldu.")

if __name__ == "__main__":
    main() 