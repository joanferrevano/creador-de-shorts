import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

# ── Añadir DLLs de CUDA ANTES de importar faster_whisper ──
def _add_cuda_dlls():
    import site, ctypes
    for sp in site.getsitepackages():
        for pkg in ["nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"]:
            path = os.path.join(sp, pkg.replace("/", os.sep))
            if os.path.isdir(path):
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
                os.add_dll_directory(path)
# ──────────────────────────────────────────────────────────

from PyQt6.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel
import google.generativeai as genai

from config import WHISPER_MODEL, GEMINI_MODEL, MIN_CLIP_SEC, MAX_CLIP_SEC


class ExtractWorker(QThread):
    log         = pyqtSignal(str)
    progress    = pyqtSignal(int, str)
    clips_ready = pyqtSignal(list)
    error       = pyqtSignal(str)
    done        = pyqtSignal()

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

        self.progress.emit(5, "Extrayendo audio...")
        self.log.emit("🎵 Extrayendo audio del vídeo...")
        audio_path = self._extract_audio(video)

        self.progress.emit(15, "Transcribiendo con Whisper (GPU)...")
        self.log.emit("🤖 Cargando Whisper large-v2 en GPU...")
        segments, duration = self._transcribe(audio_path)
        self.log.emit(f"✅ Transcripción lista — {len(segments)} segmentos, {duration:.0f}s totales")

        self.progress.emit(50, "Analizando contenido con Gemini...")
        self.log.emit("✨ Enviando transcripción a Gemini para análisis semántico...")
        clips = self._analyze_with_gemini(segments, duration)
        self.log.emit(f"🎯 Gemini identificó {len(clips)} clips potenciales")

        self.clips_ready.emit(clips)
        self.progress.emit(60, "Exportando clips con GPU (NVENC)...")
        self._export_clips(video, clips)

        os.unlink(audio_path)
        self.progress.emit(100, "¡Listo!")
        self.done.emit()

    def _extract_audio(self, video: Path) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", tmp.name
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
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
        segments = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segs
        ]
        return segments, info.duration

    def _analyze_with_gemini(self, segments, duration):
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)

        transcript = "\n".join(
            f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}"
            for s in segments
        )

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

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()

        clips = json.loads(raw).get("clips", [])

        valid_clips = []
        for c in clips:
            start = float(c["start"])
            end   = float(c["end"])
            dur   = end - start
            if dur < MIN_CLIP_SEC:
                end = start + MIN_CLIP_SEC
            elif dur > MAX_CLIP_SEC:
                end = start + MAX_CLIP_SEC
            c["start"]    = round(start, 2)
            c["end"]      = round(end,   2)
            c["duration"] = round(end - start, 2)
            valid_clips.append(c)

        return valid_clips

    def _export_clips(self, video: Path, clips: list):
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem  = video.stem
        total = len(clips)

        for i, clip in enumerate(clips):
            pct = 60 + int((i / total) * 38)
            title_safe = "".join(c for c in clip["title"] if c.isalnum() or c in " _-")[:40].strip()
            out_name   = f"{stem}_short_{i+1:02d}_{title_safe}.mp4"
            out_path   = output_dir / out_name
            clip["output_path"] = str(out_path)

            self.progress.emit(pct, f"Exportando clip {i+1}/{total}...")
            self.log.emit(f"✂️  Clip {i+1}: {clip['start']}s → {clip['end']}s — {clip['title']}")

            cmd = [
                "ffmpeg", "-y",
                "-hwaccel", "cuda",
                "-ss", str(clip["start"]),
                "-i", str(video),
                "-t", str(clip["end"] - clip["start"]),
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-rc", "vbr", "-cq", "19",
                "-b:v", "8M", "-maxrate", "12M", "-bufsize", "16M",
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