import sys
import os
import json
import subprocess
import tempfile
import threading
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QScrollArea, QFrame, QLineEdit, QSpinBox, QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette, QDragEnterEvent, QDropEvent

from faster_whisper import WhisperModel
import google.generativeai as genai



# ─── CONFIG ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # Configura la variable de entorno GEMINI_API_KEY
WHISPER_MODEL  = "large-v2"   # large-v2 = mejor calidad en español
GEMINI_MODEL   = "gemini-2.0-flash"
MIN_CLIP_SEC   = 20
MAX_CLIP_SEC   = 60
OUTPUT_SUFFIX  = "_shorts"


# ─── WORKER THREAD ────────────────────────────────────────────────────────────
class ExtractWorker(QThread):
    log       = pyqtSignal(str)
    progress  = pyqtSignal(int, str)   # value, label
    clips_ready = pyqtSignal(list)     # list of clip dicts
    error     = pyqtSignal(str)
    done      = pyqtSignal()

    def __init__(self, video_path, output_dir, num_clips, api_key):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.num_clips  = num_clips
        self.api_key    = api_key

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.error.emit(str(e))

    def _run(self):
        video = Path(self.video_path)

        # 1. Extraer audio
        self.progress.emit(5, "Extrayendo audio...")
        self.log.emit("🎵 Extrayendo audio del vídeo...")
        audio_path = self._extract_audio(video)

        # 2. Transcribir con Whisper (GPU)
        self.progress.emit(15, "Transcribiendo con Whisper (GPU)...")
        self.log.emit("🤖 Cargando Whisper large-v2 en GPU...")
        segments, duration = self._transcribe(audio_path)
        self.log.emit(f"✅ Transcripción lista — {len(segments)} segmentos, {duration:.0f}s totales")

        # 3. Analizar con Gemini
        self.progress.emit(50, "Analizando contenido con Gemini...")
        self.log.emit("✨ Enviando transcripción a Gemini para análisis semántico...")
        clips = self._analyze_with_gemini(segments, duration)
        self.log.emit(f"🎯 Gemini identificó {len(clips)} clips potenciales")

        # 4. Exportar clips con FFmpeg NVENC
        self.clips_ready.emit(clips)
        self.progress.emit(60, "Exportando clips con GPU (NVENC)...")
        self._export_clips(video, clips)

        # Limpiar audio temporal
        os.unlink(audio_path)

        self.progress.emit(100, "¡Listo!")
        self.done.emit()

    def _extract_audio(self, video: Path) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", tmp.name
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # fallback sin hwaccel para el audio
            cmd2 = ["ffmpeg", "-y", "-i", str(video),
                    "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", tmp.name]
            subprocess.run(cmd2, capture_output=True, check=True)
        return tmp.name

    def _transcribe(self, audio_path: str):
        model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
        segs, info = model.transcribe(
            audio_path,
            language="es",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )
        segments = []
        for s in segs:
            segments.append({
                "start": round(s.start, 2),
                "end":   round(s.end,   2),
                "text":  s.text.strip()
            })
        duration = info.duration
        return segments, duration

    def _analyze_with_gemini(self, segments, duration):
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Construir transcripción con timestamps
        transcript_lines = []
        for s in segments:
            transcript_lines.append(f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}")
        transcript = "\n".join(transcript_lines)

        prompt = f"""Eres un experto editor de contenido para YouTube Shorts y TikTok.
Analiza la siguiente transcripción de un vídeo de {duration:.0f} segundos y selecciona los {self.num_clips} mejores momentos para convertir en shorts virales.

CRITERIOS para elegir un buen short:
- Contenido que genere curiosidad o impacto desde el primer segundo
- Momentos con insights, consejos, historias, revelaciones o frases memorables
- Que tenga un inicio claro (sin depender del contexto anterior) y un final satisfactorio
- Duración entre {MIN_CLIP_SEC} y {MAX_CLIP_SEC} segundos
- Evita cortar en mitad de una frase o idea

TRANSCRIPCIÓN:
{transcript}

Responde ÚNICAMENTE con un JSON válido (sin texto adicional, sin markdown), con este formato exacto:
{{
  "clips": [
    {{
      "start": 12.5,
      "end": 45.0,
      "title": "Título corto del clip",
      "reason": "Por qué este momento es viral",
      "hook": "Primera frase gancho del clip"
    }}
  ]
}}

Asegúrate de que start y end sean tiempos exactos en segundos (floats) tomados del texto de la transcripción. No inventes tiempos."""

        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Limpiar posibles backticks de markdown
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()

        data = json.loads(raw)
        clips = data.get("clips", [])

        # Validar y ajustar duraciones
        valid_clips = []
        for c in clips:
            start = float(c["start"])
            end   = float(c["end"])
            dur   = end - start
            if dur < MIN_CLIP_SEC:
                end = start + MIN_CLIP_SEC
            elif dur > MAX_CLIP_SEC:
                end = start + MAX_CLIP_SEC
            c["start"] = round(start, 2)
            c["end"]   = round(end,   2)
            c["duration"] = round(end - start, 2)
            valid_clips.append(c)

        return valid_clips

    def _export_clips(self, video: Path, clips: list):
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = video.stem

        total = len(clips)
        for i, clip in enumerate(clips):
            pct = 60 + int((i / total) * 38)
            title_safe = "".join(c for c in clip["title"] if c.isalnum() or c in " _-")[:40].strip()
            out_name = f"{stem}_short_{i+1:02d}_{title_safe}.mp4"
            out_path = output_dir / out_name
            clip["output_path"] = str(out_path)

            self.progress.emit(pct, f"Exportando clip {i+1}/{total}...")
            self.log.emit(f"✂️  Clip {i+1}: {clip['start']}s → {clip['end']}s — {clip['title']}")

            duration = clip["end"] - clip["start"]
            cmd = [
                "ffmpeg", "-y",
                "-hwaccel", "cuda",
                "-ss", str(clip["start"]),
                "-i", str(video),
                "-t", str(duration),
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-rc", "vbr",
                "-cq", "19",
                "-b:v", "8M",
                "-maxrate", "12M",
                "-bufsize", "16M",
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                "-r", "60",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(out_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.log.emit(f"⚠️  Error en clip {i+1}: {result.stderr[-200:]}")
            else:
                size_mb = out_path.stat().st_size / 1024 / 1024
                self.log.emit(f"   ✅ Guardado: {out_name} ({size_mb:.1f} MB)")


# ─── CLIP CARD WIDGET ─────────────────────────────────────────────────────────
class ClipCard(QFrame):
    def __init__(self, clip: dict, index: int):
        super().__init__()
        self.clip = clip
        self.setObjectName("clipCard")
        self.setStyleSheet("""
            #clipCard {
                background: #1a1a2e;
                border: 1px solid #2d2d4e;
                border-radius: 10px;
                padding: 4px;
            }
            #clipCard:hover { border: 1px solid #6c63ff; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Header
        header = QHBoxLayout()
        num = QLabel(f"#{index}")
        num.setStyleSheet("color: #6c63ff; font-weight: bold; font-size: 16px;")
        title = QLabel(clip.get("title", "Sin título"))
        title.setStyleSheet("color: #e0e0ff; font-weight: bold; font-size: 13px;")
        title.setWordWrap(True)
        dur = QLabel(f"{clip.get('duration', 0):.0f}s")
        dur.setStyleSheet("color: #a0a0c0; font-size: 12px;")
        header.addWidget(num)
        header.addWidget(title, 1)
        header.addWidget(dur)
        layout.addLayout(header)

        # Timestamps
        ts = QLabel(f"⏱ {clip.get('start', 0):.1f}s → {clip.get('end', 0):.1f}s")
        ts.setStyleSheet("color: #6c63ff; font-size: 11px;")
        layout.addWidget(ts)

        # Hook
        hook = QLabel(f"🎣 {clip.get('hook', '')}")
        hook.setStyleSheet("color: #c0c0e0; font-size: 11px; font-style: italic;")
        hook.setWordWrap(True)
        layout.addWidget(hook)

        # Reason
        reason = QLabel(clip.get("reason", ""))
        reason.setStyleSheet("color: #808099; font-size: 11px;")
        reason.setWordWrap(True)
        layout.addWidget(reason)

        # Output path (if available)
        if clip.get("output_path"):
            path_label = QLabel(f"📁 {Path(clip['output_path']).name}")
            path_label.setStyleSheet("color: #4caf82; font-size: 10px;")
            path_label.setWordWrap(True)
            layout.addWidget(path_label)


# ─── MAIN WINDOW ──────────────────────────────────────────────────────────────
class ShortsExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shorts Extractor — AI powered")
        self.setMinimumSize(900, 700)
        self.resize(1100, 750)
        self.video_path = None
        self.output_dir = None
        self.worker = None
        self._apply_theme()
        self._build_ui()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d0d1a;
                color: #e0e0ff;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton#primary {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6c63ff, stop:1 #a855f7);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton#primary:hover { background: #7c73ff; }
            QPushButton#primary:disabled { background: #333355; color: #666688; }
            QPushButton#secondary {
                background: #1a1a2e;
                color: #a0a0d0;
                border: 1px solid #2d2d4e;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 12px;
            }
            QPushButton#secondary:hover { border-color: #6c63ff; color: #e0e0ff; }
            QLineEdit, QSpinBox {
                background: #1a1a2e;
                border: 1px solid #2d2d4e;
                border-radius: 6px;
                padding: 6px 10px;
                color: #e0e0ff;
                font-size: 12px;
            }
            QLineEdit:focus, QSpinBox:focus { border-color: #6c63ff; }
            QProgressBar {
                background: #1a1a2e;
                border: 1px solid #2d2d4e;
                border-radius: 6px;
                height: 16px;
                text-align: center;
                color: white;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6c63ff, stop:1 #a855f7);
                border-radius: 6px;
            }
            QTextEdit {
                background: #0a0a15;
                border: 1px solid #1d1d35;
                border-radius: 8px;
                color: #a0ffa0;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 8px;
            }
            QScrollArea { border: none; background: transparent; }
            QLabel#sectionTitle {
                color: #6c63ff;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QLabel#apiKeyLabel { color: #808099; font-size: 11px; }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(16)
        root.setContentsMargins(20, 20, 20, 20)

        # ── LEFT PANEL ──
        left = QVBoxLayout()
        left.setSpacing(14)
        root.addLayout(left, 2)

        # Logo / title
        title_label = QLabel("⚡ SHORTS EXTRACTOR")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #e0e0ff; letter-spacing: 2px;")
        sub = QLabel("Extrae clips virales con IA + Whisper + NVENC")
        sub.setStyleSheet("color: #6060a0; font-size: 11px;")
        left.addWidget(title_label)
        left.addWidget(sub)

        # Separator
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #1d1d35;")
        left.addWidget(sep)

        # API Key
        lbl_api = QLabel("GEMINI API KEY"); lbl_api.setObjectName("sectionTitle")
        left.addWidget(lbl_api)
        self.api_key_input = QLineEdit(GEMINI_API_KEY)
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("AIza...")
        left.addWidget(self.api_key_input)

        # Video file
        lbl_vid = QLabel("VÍDEO DE ENTRADA"); lbl_vid.setObjectName("sectionTitle")
        left.addWidget(lbl_vid)
        file_row = QHBoxLayout()
        self.file_label = QLabel("Ningún archivo seleccionado")
        self.file_label.setStyleSheet("color: #606080; font-size: 11px;")
        self.file_label.setWordWrap(True)
        btn_file = QPushButton("Elegir vídeo"); btn_file.setObjectName("secondary")
        btn_file.clicked.connect(self._pick_video)
        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(btn_file)
        left.addLayout(file_row)

        # Output dir
        lbl_out = QLabel("CARPETA DE SALIDA"); lbl_out.setObjectName("sectionTitle")
        left.addWidget(lbl_out)
        out_row = QHBoxLayout()
        self.out_label = QLabel("Misma carpeta que el vídeo")
        self.out_label.setStyleSheet("color: #606080; font-size: 11px;")
        btn_out = QPushButton("Cambiar"); btn_out.setObjectName("secondary")
        btn_out.clicked.connect(self._pick_output)
        out_row.addWidget(self.out_label, 1)
        out_row.addWidget(btn_out)
        left.addLayout(out_row)

        # Num clips
        lbl_clips = QLabel("NÚMERO DE CLIPS A GENERAR"); lbl_clips.setObjectName("sectionTitle")
        left.addWidget(lbl_clips)
        self.num_clips_spin = QSpinBox()
        self.num_clips_spin.setRange(1, 20)
        self.num_clips_spin.setValue(5)
        left.addWidget(self.num_clips_spin)

        # Progress
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        left.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left.addWidget(self.progress_bar)

        # Main button
        self.btn_run = QPushButton("🚀  Analizar y Extraer Clips")
        self.btn_run.setObjectName("primary")
        self.btn_run.setFixedHeight(44)
        self.btn_run.clicked.connect(self._start)
        left.addWidget(self.btn_run)

        # Open output folder button
        self.btn_open = QPushButton("📂  Abrir carpeta de salida")
        self.btn_open.setObjectName("secondary")
        self.btn_open.setVisible(False)
        self.btn_open.clicked.connect(self._open_output)
        left.addWidget(self.btn_open)

        left.addStretch()

        # Log
        lbl_log = QLabel("LOG"); lbl_log.setObjectName("sectionTitle")
        left.addWidget(lbl_log)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(180)
        left.addWidget(self.log_box)

        # ── RIGHT PANEL ──
        right = QVBoxLayout()
        right.setSpacing(10)
        root.addLayout(right, 3)

        lbl_clips_title = QLabel("CLIPS DETECTADOS"); lbl_clips_title.setObjectName("sectionTitle")
        right.addWidget(lbl_clips_title)

        self.clips_scroll = QScrollArea()
        self.clips_scroll.setWidgetResizable(True)
        self.clips_container = QWidget()
        self.clips_layout = QVBoxLayout(self.clips_container)
        self.clips_layout.setSpacing(10)
        self.clips_layout.addStretch()
        self.clips_scroll.setWidget(self.clips_container)
        right.addWidget(self.clips_scroll)

        # Empty state
        self.empty_label = QLabel("Los clips aparecerán aquí\ndespués del análisis")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #2d2d4e; font-size: 16px;")
        self.clips_layout.insertWidget(0, self.empty_label)

    def _pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar vídeo", "",
            "Vídeos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)"
        )
        if path:
            self.video_path = path
            name = Path(path).name
            self.file_label.setText(name)
            self.file_label.setStyleSheet("color: #a0ffa0; font-size: 11px;")
            if not self.output_dir:
                self.out_label.setText(str(Path(path).parent))

    def _pick_output(self):
        path = QFileDialog.getExistingDirectory(self, "Carpeta de salida")
        if path:
            self.output_dir = path
            self.out_label.setText(path)
            self.out_label.setStyleSheet("color: #a0ffa0; font-size: 11px;")

    def _start(self):
        if not self.video_path:
            QMessageBox.warning(self, "Sin vídeo", "Selecciona un vídeo primero.")
            return

        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Sin API Key", "Introduce tu Gemini API Key.")
            return

        output_dir = self.output_dir or str(Path(self.video_path).parent / "shorts")
        self.current_output_dir = output_dir

        # Limpiar clips anteriores
        while self.clips_layout.count() > 1:
            item = self.clips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.empty_label.setText("Analizando...")
        self.empty_label.setStyleSheet("color: #6060a0; font-size: 16px;")

        self.btn_run.setEnabled(False)
        self.btn_open.setVisible(False)
        self.log_box.clear()
        self.progress_bar.setValue(0)

        self.worker = ExtractWorker(
            self.video_path, output_dir,
            self.num_clips_spin.value(), api_key
        )
        self.worker.log.connect(self._on_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.clips_ready.connect(self._show_clips)
        self.worker.error.connect(self._on_error)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _on_log(self, msg):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    def _on_progress(self, val, label):
        self.progress_bar.setValue(val)
        self.progress_label.setText(label)

    def _show_clips(self, clips):
        # Remove empty label
        self.empty_label.setVisible(False)
        for i, clip in enumerate(clips):
            card = ClipCard(clip, i + 1)
            self.clips_layout.insertWidget(i, card)

    def _on_error(self, msg):
        self.log_box.append(f"❌ ERROR: {msg}")
        self.btn_run.setEnabled(True)
        self.progress_label.setText("Error")
        QMessageBox.critical(self, "Error", msg)

    def _on_done(self):
        self.btn_run.setEnabled(True)
        self.btn_open.setVisible(True)
        self._on_log("🎉 ¡Proceso completado!")

    def _open_output(self):
        if hasattr(self, "current_output_dir"):
            os.startfile(self.current_output_dir)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ShortsExtractor()
    window.show()
    sys.exit(app.exec())