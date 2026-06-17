import os
import sys

# Añadir DLLs de CUDA al PATH ANTES de cualquier import
_SITE = r"C:\Users\Joanf\AppData\Local\Programs\Python\Python310\Lib\site-packages"
_CUDA_PATHS = [
    os.path.join(_SITE, "nvidia", "cublas", "bin"),
    os.path.join(_SITE, "nvidia", "cudnn", "bin"),
    os.path.join(_SITE, "nvidia", "cuda_nvrtc", "bin"),
]
for p in _CUDA_PATHS:
    if os.path.isdir(p):
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(p)

# Ahora sí arrancamos la app
from main import ShortsExtractor
from PyQt6.QtWidgets import QApplication
app = QApplication(sys.argv)
app.setStyle("Fusion")
window = ShortsExtractor()
window.show()
sys.exit(app.exec())
