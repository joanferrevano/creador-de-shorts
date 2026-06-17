import sys
import os
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QScrollArea, QFrame, QLineEdit, QSpinBox, QMessageBox
)
from PyQt6.QtCore import Qt

from config import GEMINI_API_KEY
from worker import ExtractWorker


# ─── CLIP CARD ────────────────────────────────────────────────────────────────
class ClipCard(QFrame):
    def __init__(self, clip: dict, index: int):
        super().__init__()
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

        ts = QLabel(f"⏱ {clip.get('start', 0):.1f}s → {clip.get('end', 0):.1f}s")
        ts.setStyleSheet("color: #6c63ff; font-size: 11px;")
        layout.addWidget(ts)

        hook = QLabel(f"🎣 {clip.get('hook', '')}")
        hook.setStyleSheet("color: #c0c0e0; font-size: 11px; font-style: italic;")
        hook.setWordWrap(True)
        layout.addWidget(hook)

        reason = QLabel(clip.get("reason", ""))
        reason.setStyleSheet("color: #808099; font-size: 11px;")
        reason.setWordWrap(True)
        layout.addWidget(reason)

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
        self.worker     = None
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
                color: white; border: none; border-radius: 8px;
                padding: 10px 24px; font-size: 13px; font-weight: bold;
            }
            QPushButton#primary:hover    { background: #7c73ff; }
            QPushButton#primary:disabled { background: #333355; color: #666688; }
            QPushButton#secondary {
                background: #1a1a2e; color: #a0a0d0;
                border: 1px solid #2d2d4e; border-radius: 8px;
                padding: 8px 16px; font-size: 12px;
            }
            QPushButton#secondary:hover { border-color: #6c63ff; color: #e0e0ff; }
            QLineEdit, QSpinBox {
                background: #1a1a2e; border: 1px solid #2d2d4e;
                border-radius: 6px; padding: 6px 10px;
                color: #e0e0ff; font-size: 12px;
            }
            QLineEdit:focus, QSpinBox:focus { border-color: #6c63ff; }
            QProgressBar {
                background: #1a1a2e; border: 1px solid #2d2d4e;
                border-radius: 6px; height: 16px;
                text-align: center; color: white; font-size: 11px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6c63ff, stop:1 #a855f7);
                border-radius: 6px;
            }
            QTextEdit {
                background: #0a0a15; border: 1px solid #1d1d35;
                border-radius: 8px; color: #a0ffa0;
                font-family: 'Consolas', monospace;
                font-size: 11px; padding: 8px;
            }
            QScrollArea { border: none; background: transparent; }
            QLabel#sectionTitle {
                color: #6c63ff; font-size: 11px;
                font-weight: bold; letter-spacing: 1px;
            }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(16)
        root.setContentsMargins(20, 20, 20, 20)

        # ── LEFT ──
        left = QVBoxLayout()
        left.setSpacing(14)
        root.addLayout(left, 2)

        title_label = QLabel("⚡ SHORTS EXTRACTOR")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #e0e0ff; letter-spacing: 2px;")
        sub = QLabel("Extrae clips virales con IA + Whisper + NVENC")
        sub.setStyleSheet("color: #6060a0; font-size: 11px;")
        left.addWidget(title_label)
        left.addWidget(sub)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #1d1d35;")
        left.addWidget(sep)

        lbl_api = QLabel("GEMINI API KEY"); lbl_api.setObjectName("sectionTitle")
        left.addWidget(lbl_api)
        self.api_key_input = QLineEdit(GEMINI_API_KEY)
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("AIza...")
        left.addWidget(self.api_key_input)

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

        lbl_n = QLabel("NÚMERO DE CLIPS A GENERAR"); lbl_n.setObjectName("sectionTitle")
        left.addWidget(lbl_n)
        self.num_clips_spin = QSpinBox()
        self.num_clips_spin.setRange(1, 20)
        self.num_clips_spin.setValue(5)
        left.addWidget(self.num_clips_spin)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        left.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left.addWidget(self.progress_bar)

        self.btn_run = QPushButton("🚀  Analizar y Extraer Clips")
        self.btn_run.setObjectName("primary")
        self.btn_run.setFixedHeight(44)
        self.btn_run.clicked.connect(self._start)
        left.addWidget(self.btn_run)

        self.btn_open = QPushButton("📂  Abrir carpeta de salida")
        self.btn_open.setObjectName("secondary")
        self.btn_open.setVisible(False)
        self.btn_open.clicked.connect(self._open_output)
        left.addWidget(self.btn_open)

        left.addStretch()

        lbl_log = QLabel("LOG"); lbl_log.setObjectName("sectionTitle")
        left.addWidget(lbl_log)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(180)
        left.addWidget(self.log_box)

        # ── RIGHT ──
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

        self.empty_label = QLabel("Los clips aparecerán aquí\ndespués del análisis")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #2d2d4e; font-size: 16px;")
        self.clips_layout.insertWidget(0, self.empty_label)

    # ── Slots ──────────────────────────────────────────────────────────────────
    def _pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar vídeo", "",
            "Vídeos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)"
        )
        if path:
            self.video_path = path
            self.file_label.setText(Path(path).name)
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

        while self.clips_layout.count() > 1:
            item = self.clips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.empty_label.setText("Analizando...")
        self.empty_label.setStyleSheet("color: #6060a0; font-size: 16px;")
        self.empty_label.setVisible(True)

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
        self.empty_label.setVisible(False)
        for i, clip in enumerate(clips):
            self.clips_layout.insertWidget(i, ClipCard(clip, i + 1))

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
