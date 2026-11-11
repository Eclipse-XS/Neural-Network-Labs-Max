import os
import json
import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image

class GalleryService:
    """Persistent storage engine for Multimodal Studio records using SQLite and local file storage."""
    
    def __init__(self, storage_dir: Optional[Path] = None, db_name: str = "gallery.db"):
        if storage_dir is None:
            self.storage_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "gallery"
        else:
            self.storage_dir = Path(storage_dir)
            
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / db_name
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS studio_records (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    prompt TEXT,
                    parameters_json TEXT,
                    input_image_path TEXT,
                    output_image_path TEXT,
                    output_text TEXT,
                    latency_sec REAL,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
            """)
            conn.commit()

    def add_record(
        self,
        task_type: str,
        prompt: str,
        parameters: Dict[str, Any],
        output_text: str = "",
        input_image: Optional[Image.Image] = None,
        output_image: Optional[Image.Image] = None,
        latency_sec: float = 0.0,
        status: str = "success",
        error_message: Optional[str] = None
    ) -> str:
        """Save a new generation/caption/VQA record persistently."""
        record_id = f"{task_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]}"
        timestamp = datetime.datetime.now().isoformat()
        
        input_image_rel = None
        output_image_rel = None
        
        if input_image is not None and status != "failed":
            in_filename = f"{record_id}_input.png"
            in_path = self.storage_dir / in_filename
            input_image.save(in_path)
            input_image_rel = in_filename
            
        if output_image is not None and status != "failed":
            out_filename = f"{record_id}_output.png"
            out_path = self.storage_dir / out_filename
            output_image.save(out_path)
            output_image_rel = out_filename
            
        params_str = json.dumps(parameters, ensure_ascii=False)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO studio_records (
                    id, timestamp, task_type, prompt, parameters_json,
                    input_image_path, output_image_path, output_text, latency_sec, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_id, timestamp, task_type, prompt, params_str,
                input_image_rel, output_image_rel, output_text, latency_sec, status, error_message
            ))
            conn.commit()
            
        return record_id

    def get_all_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent generation records sorted by timestamp descending."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM studio_records ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            
        records = []
        for r in rows:
            d = dict(r)
            if d.get("parameters_json"):
                try:
                    d["parameters"] = json.loads(d["parameters_json"])
                except Exception:
                    d["parameters"] = {}
            records.append(d)
        return records

    get_records = get_all_records

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific record by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM studio_records WHERE id = ?", (record_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            if d.get("parameters_json"):
                try:
                    d["parameters"] = json.loads(d["parameters_json"])
                except Exception:
                    d["parameters"] = {}
            return d
