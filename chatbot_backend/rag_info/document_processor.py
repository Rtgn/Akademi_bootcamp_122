import os
import json
from pathlib import Path
from typing import List, Dict, Any
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

class DocumentProcessor:
    def __init__(self, documents_dir: str = "rag_pregnancy_reports"):
        self.documents_dir = Path(documents_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def extract_text_from_docx(self, file_path: Path) -> str:
        """DOCX dosyasından metin çıkarır"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            print(f"Hata: {file_path} dosyası okunamadı: {e}")
            return ""
    
    def process_documents(self) -> List[Dict[str, Any]]:
        """Tüm belgeleri işler ve chunk'lara böler"""
        all_chunks = []
        
        # DOCX dosyalarını bul
        docx_files = list(self.documents_dir.glob("*.docx"))
        
        print(f"Toplam {len(docx_files)} DOCX dosyası bulundu.")
        
        for file_path in docx_files:
            print(f"İşleniyor: {file_path.name}")
            
            text = self.extract_text_from_docx(file_path)
            
            if not text.strip():
                print(f"Uyarı: {file_path.name} dosyası boş veya okunamadı")
                continue
            
            chunks = self.text_splitter.split_text(text)
            
            # Her chunk için metadata oluştur
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "id": self._generate_chunk_id(file_path.name, i),
                    "content": chunk,
                    "metadata": {
                        "source_file": file_path.name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_size": file_path.stat().st_size,
                        "chunk_size": len(chunk)
                    }
                }
                all_chunks.append(chunk_data)
            
            print(f"  - {len(chunks)} chunk oluşturuldu")
        
        return all_chunks
    
    def _generate_chunk_id(self, filename: str, chunk_index: int) -> str:
        content = f"{filename}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def save_chunks_to_json(self, chunks: List[Dict[str, Any]], output_file: str = "processed_chunks.json"):
        """Chunk'ları JSON dosyasına kaydeder"""
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"Chunk'lar {output_path} dosyasına kaydedildi.")
        print(f"Toplam {len(chunks)} chunk işlendi.")
    
    def save_chunks_to_jsonl(self, chunks: List[Dict[str, Any]], output_file: str = "processed_chunks.jsonl"):
        """Chunk'ları JSONL dosyasına kaydeder"""
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        print(f"Chunk'lar {output_path} dosyasına kaydedildi.")
        print(f"Toplam {len(chunks)} chunk işlendi.")

def main():
    processor = DocumentProcessor()
    
    print("Belgeler işleniyor...")
    chunks = processor.process_documents()
    
    if chunks:
        # JSON formatında kaydet
        processor.save_chunks_to_json(chunks)
        
        # JSONL formatında da kaydet (RAG sistemleri için daha uygun)
        processor.save_chunks_to_jsonl(chunks)
        
        # İstatistikler
        total_content_length = sum(len(chunk["content"]) for chunk in chunks)
        avg_chunk_size = total_content_length / len(chunks)
        
        print(f"\nİstatistikler:")
        print(f"- Toplam chunk sayısı: {len(chunks)}")
        print(f"- Toplam karakter sayısı: {total_content_length:,}")
        print(f"- Ortalama chunk boyutu: {avg_chunk_size:.0f} karakter")
        
    else:
        print("Hiç chunk oluşturulamadı!")

if __name__ == "__main__":
    main() 