import json
import torch
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGSystem:
    def __init__(self, 
                 llm_model_path: str = "stabilityai/stablelm-2-zephyr-1_6b",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_db_path: str = None):
       
        if vector_db_path is None:
            # Mevcut dosyanın bulunduğu dizini al
            current_file_dir = Path(__file__).parent
            self.vector_db_path = current_file_dir / "vector_database"
        else:
            self.vector_db_path = Path(vector_db_path)
        
        self.llm_model_path = llm_model_path
        
        print("RAG sistemi başlatılıyor...")
        print(f"Vektör veritabanı yolu: {self.vector_db_path}")
        
        # Embedding modelini yükle
        print("Embedding modeli yükleniyor...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # LLM modelini yükle
        print("LLM modeli yükleniyor...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # FAISS indeksini yükle
        print("Vektör indeksi yükleniyor...")
        faiss_index_path = self.vector_db_path / "faiss_index.bin"
        if not faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS indeks dosyası bulunamadı: {faiss_index_path}")
        
        self.index = faiss.read_index(str(faiss_index_path))
        
        # Chunk metadata'larını yükle
        chunks_metadata_path = self.vector_db_path / "chunks_metadata.json"
        if not chunks_metadata_path.exists():
            raise FileNotFoundError(f"Chunks metadata dosyası bulunamadı: {chunks_metadata_path}")
        
        with open(chunks_metadata_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print("RAG sistemi hazır!")
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Sorgu embedding'i oluştur
        query_embedding = self.embedding_model.encode([query])
        
        # En yakın vektörleri bul
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Sonuçları formatla
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            chunk = self.chunks[idx]
            results.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "similarity_score": float(distance)
            })
        
        return results
    
    def create_context(self, similar_chunks: List[Dict[str, Any]]) -> str:
        """Benzer chunk'lardan context oluşturur"""
        context_parts = []
        
        for i, chunk in enumerate(similar_chunks, 1):
            context_parts.append(f"Source {i} (File: {chunk['metadata']['source_file']}):")
            context_parts.append(chunk['content'])
            context_parts.append("")  # Boş satır
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str, max_new_tokens: int = 256) -> str:
        """LLM ile yanıt üretir"""
        # Prompt template oluştur
        prompt = f"""<|system|>
You are an expert health consultant specializing in pregnancy and postpartum care.
Use the following information to answer questions. Provide complete, helpful responses.
Never mention sources, references, or file names in your response.
Always give complete answers without cutting off mid-sentence.

{context}

<|user|>
{query}

<|assistant|>"""
        
        # Tokenize et
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        # Yanıt üret
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', None),
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Yanıtı decode et
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Prompt'u çıkar, sadece yanıtı al
        response = response.replace(prompt, "").strip()
        
        return response
    
    def generate_response_streaming(self, query: str, context: str, max_new_tokens: int = 256):
        """LLM ile yanıt üretir (streaming)"""
        # Prompt template oluştur
        prompt = f"""<|system|>
You are an expert health consultant specializing in pregnancy and postpartum care.
Use the following information to answer questions. Provide complete, helpful responses.
Never mention sources, references, or file names in your response.
Always give complete answers without cutting off mid-sentence.

{context}

<|user|>
{query}

<|assistant|>"""
        
        # Tokenize et
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        # Streaming yanıt üret
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask', None),
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.1
                )
                
                # Yeni token'ı al
                new_token = outputs[0][-1:]
                
                # EOS token kontrolü
                if new_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Token'ı decode et ve yield et
                new_text = self.tokenizer.decode(new_token, skip_special_tokens=True)
                if new_text.strip():
                    yield new_text
                
                # Input'ları güncelle
                inputs['input_ids'] = outputs
                # Update attention mask for next iteration
                if 'attention_mask' in inputs:
                    attention_mask = inputs['attention_mask']
                    inputs['attention_mask'] = torch.cat([attention_mask, torch.ones(1, 1, device=attention_mask.device)], dim=1)
    
    def answer_question(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Ana RAG fonksiyonu - soruyu yanıtlar"""
        print(f"Question: {query}")
        
        # 1. Benzer chunk'ları ara
        print("Searching for similar documents...")
        similar_chunks = self.search_similar_chunks(query, top_k)
        
        # 2. Context oluştur
        context = self.create_context(similar_chunks)
        
        # 3. LLM ile yanıt üret
        print("Generating response...")
        response = self.generate_response(query, context)
        
        # 4. Sonuçları formatla
        result = {
            "query": query,
            "response": response,
            "sources": [
                {
                    "file": chunk["metadata"]["source_file"],
                    "similarity_score": chunk["similarity_score"],
                    "content_preview": chunk["content"][:200] + "..."
                }
                for chunk in similar_chunks
            ],
            "context_length": len(context)
        }
        
        return result
    
    def answer_question_streaming(self, query: str, top_k: int = 5):
        """Ana RAG fonksiyonu - soruyu yanıtlar (streaming)"""
        print(f"Question: {query}")
        
        # 1. Benzer chunk'ları ara
        print("Searching for similar documents...")
        similar_chunks = self.search_similar_chunks(query, top_k)
        
        # 2. Context oluştur
        context = self.create_context(similar_chunks)
        
        # 3. LLM ile streaming yanıt üret
        print("Generating response...")
        print("Answer: ", end="", flush=True)
        
        response_text = ""
        for token in self.generate_response_streaming(query, context):
            print(token, end="", flush=True)
            response_text += token
            import time
            time.sleep(0.01)  # Harf harf yazma hızı
        
        print()  # Yeni satır
        
        result = {
            "query": query,
            "response": response_text,
            "sources": [
                {
                    "file": chunk["metadata"]["source_file"],
                    "similarity_score": chunk["similarity_score"],
                    "content_preview": chunk["content"][:200] + "..."
                }
                for chunk in similar_chunks
            ],
            "context_length": len(context)
        }
        
        return result
    
    def interactive_chat(self):
        print("\n" + "="*50)
        print("RAG System - Interactive Chat (Streaming Mode)")
        print("Type 'quit' or 'exit' to exit")
        print("="*50)
        
        while True:
            try:
                query = input("\nQuestion: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    print("Ending chat...")
                    break
                
                if not query:
                    continue
                
                # Streaming yanıt üret
                result = self.answer_question_streaming(query)
                
            except KeyboardInterrupt:
                print("\nEnding chat...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    # RAG sistemini başlat
    rag = RAGSystem()
    
    rag.interactive_chat()

if __name__ == "__main__":
    main() 