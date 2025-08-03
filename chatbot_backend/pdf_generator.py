import os
import sqlite3
import uuid
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import json
from typing import Dict, Any, Optional

class PDFGenerator:
    def __init__(self, db_path: str = "nutrition_programs.db"):
        """PDF oluşturucu sınıfını başlatır"""
        self.db_path = db_path
        self.programs_dir = "nutrition_programs"
        self._init_database()
        self._ensure_programs_directory()
    
    def _init_database(self):
        """Veritabanını başlatır ve tabloları oluşturur"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Kullanıcı programları tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_programs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL,
                program_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                program_type TEXT DEFAULT 'nutrition',
                status TEXT DEFAULT 'active'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _ensure_programs_directory(self):
        """Programlar klasörünün varlığını kontrol eder ve oluşturur"""
        if not os.path.exists(self.programs_dir):
            os.makedirs(self.programs_dir)
    
    def create_nutrition_pdf(self, user_id: str, username: str, program_content: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Beslenme programı PDF'i oluşturur ve veritabanına kaydeder"""
        try:
            # Benzersiz program ID'si oluştur
            program_id = str(uuid.uuid4())
            
            # Dosya adı oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nutrition_program_{username}_{timestamp}.pdf"
            file_path = os.path.join(self.programs_dir, filename)
            
            # PDF oluştur
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Başlık
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph('Kişiselleştirilmiş Beslenme Programı', title_style))
            story.append(Spacer(1, 20))
            
            # Kullanıcı bilgileri
            story.append(Paragraph(f'<b>Hazırlayan:</b> {username}', styles['Normal']))
            story.append(Paragraph(f'<b>Oluşturulma Tarihi:</b> {datetime.now().strftime("%d/%m/%Y %H:%M")}', styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Kullanıcı verileri tablosu
            user_info_data = [
                ['Kişisel Bilgiler', ''],
                ['Yaş', str(user_data.get('age', 'Belirtilmemiş'))],
                ['Kilo (kg)', str(user_data.get('weight', 'Belirtilmemiş'))],
                ['Boy (cm)', str(user_data.get('height', 'Belirtilmemiş'))],
                ['Aktivite Seviyesi', str(user_data.get('activity_level', 'Belirtilmemiş'))],
                ['Meslek', str(user_data.get('occupation', 'Belirtilmemiş'))],
                ['Uyku Saati', str(user_data.get('sleep_hours', 'Belirtilmemiş'))],
                ['Stres Seviyesi', str(user_data.get('stress_level', 'Belirtilmemiş'))],
            ]
            
            # Hamilelik bilgileri
            if user_data.get('pregnancy_status') == 'yes':
                user_info_data.extend([
                    ['', ''],
                    ['Hamilelik Bilgileri', ''],
                    ['Hamilelik Haftası', str(user_data.get('pregnancy_weeks', 'Belirtilmemiş'))],
                ])
            
            # Doğum sonrası bilgileri
            if user_data.get('postpartum_weeks'):
                user_info_data.extend([
                    ['', ''],
                    ['Doğum Sonrası Bilgileri', ''],
                    ['Doğum Sonrası Hafta', str(user_data.get('postpartum_weeks', 'Belirtilmemiş'))],
                    ['Doğum Yöntemi', str(user_data.get('delivery_method', 'Belirtilmemiş'))],
                    ['Emzirme', str(user_data.get('breastfeeding', 'Belirtilmemiş'))],
                ])
                if user_data.get('breastfeeding') == 'yes':
                    user_info_data.append(['Emzirme Haftası', str(user_data.get('breastfeeding_weeks', 'Belirtilmemiş'))])
            
            # Sağlık bilgileri
            user_info_data.extend([
                ['', ''],
                ['Sağlık Bilgileri', ''],
                ['Alerjiler', str(user_data.get('allergies', 'Yok'))],
                ['Tıbbi Durumlar', str(user_data.get('medical_conditions', 'Yok'))],
                ['İlaçlar', str(user_data.get('medications', 'Yok'))],
                ['Sindirim Sorunları', str(user_data.get('digestive_issues', 'Yok'))],
            ])
            
            # Beslenme tercihleri
            user_info_data.extend([
                ['', ''],
                ['Beslenme Tercihleri', ''],
                ['Diyet Tercihleri', str(user_data.get('dietary_preferences', 'Yok'))],
                ['Sevilmeyen Yiyecekler', str(user_data.get('food_dislikes', 'Yok'))],
                ['Pişirme Becerileri', str(user_data.get('cooking_skills', 'Belirtilmemiş'))],
                ['Hedefler', str(user_data.get('goals', 'Belirtilmemiş'))],
            ])
            
            # Tablo oluştur
            user_info_table = Table(user_info_data, colWidths=[2*inch, 4*inch])
            user_info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(user_info_table)
            story.append(Spacer(1, 30))
            
            # Program içeriği
            story.append(Paragraph('<b>Beslenme Programı:</b>', styles['Heading2']))
            story.append(Spacer(1, 20))
            
            # Program içeriğini paragraflar halinde ekle
            paragraphs = program_content.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    # Türkçe karakterleri düzelt
                    paragraph = paragraph.replace('\\n', '\n').replace('\\t', '\t')
                    story.append(Paragraph(paragraph, styles['Normal']))
                    story.append(Spacer(1, 12))
            
            # PDF'i oluştur
            doc.build(story)
            
            # Veritabanına kaydet
            self._save_to_database(user_id, username, program_id, filename, file_path)
            
            return {
                "success": True,
                "program_id": program_id,
                "filename": filename,
                "file_path": file_path,
                "message": "Beslenme programı PDF olarak oluşturuldu ve kaydedildi."
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"PDF oluşturma hatası: {str(e)}")
            print(f"Hata detayları: {error_details}")
            return {
                "success": False,
                "error": str(e),
                "error_details": error_details,
                "message": "PDF oluşturulurken hata oluştu."
            }
    

    
    def _save_to_database(self, user_id: str, username: str, program_id: str, filename: str, file_path: str):
        """Program bilgilerini veritabanına kaydeder"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_programs (user_id, username, program_id, filename, file_path, program_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, username, program_id, filename, file_path, 'nutrition'))
        
        conn.commit()
        conn.close()
    
    def get_user_programs(self, user_id: str) -> list:
        """Kullanıcının programlarını getirir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT program_id, filename, created_at, program_type, status
            FROM user_programs
            WHERE user_id = ? AND status = 'active'
            ORDER BY created_at DESC
        ''', (user_id,))
        
        programs = []
        for row in cursor.fetchall():
            programs.append({
                "program_id": row[0],
                "filename": row[1],
                "created_at": row[2],
                "program_type": row[3],
                "status": row[4]
            })
        
        conn.close()
        return programs
    
    def get_program_file_path(self, program_id: str) -> Optional[str]:
        """Program ID'sine göre dosya yolunu getirir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT file_path
            FROM user_programs
            WHERE program_id = ? AND status = 'active'
        ''', (program_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def delete_program(self, program_id: str, user_id: str) -> bool:
        """Programı siler (soft delete)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE user_programs
            SET status = 'deleted'
            WHERE program_id = ? AND user_id = ?
        ''', (program_id, user_id))
        
        affected_rows = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected_rows > 0 