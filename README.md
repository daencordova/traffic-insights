# 🚗 Traffic Insights — Sistema de Reconocimiento de Tráfico en Tiempo Real

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV 4.8+](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.15%2B-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Status: Production](https://img.shields.io/badge/status-production-brightgreen.svg)](https://github.com/daencordova/traffic-insights)

**Sistema de seguimiento de vehículos en tiempo real con YOLOv8 y tracking avanzado**

</div>

---

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [Controles](#-controles)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Documentación Técnica](#-documentación-técnica)
- [Rendimiento](#-rendimiento)
- [Solución de Problemas](#-solución-de-problemas)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)

---

## ✨ Características

### 🎯 Detección de Objetos
- **YOLOv8** optimizado con soporte para ONNX Runtime
- Detección de vehículos: automóviles, motocicletas, autobuses, camiones
- Preprocesamiento de imagen (reducción de ruido, ecualización de histograma)
- Caché LRU de detecciones para mejorar rendimiento
- Soporte para inferencia por lotes (batch inference)

### 🎯 Tracking Avanzado
- **Tracker híbrido** con re-identificación de objetos
- Filtro de Kalman optimizado con Numba para tracking suave
- Matching jerárquico (IoU + features visuales + movimiento)
- Sistema de hipótesis múltiples (MHT) para manejo de oclusiones
- Fusión de sensores para tracking multi-modal
- Predicción de trayectoria con modelos adaptativos
- Aprendizaje en línea para adaptación de apariencia

### 📊 Visualización y Dashboard
- FPS y tiempo de procesamiento en tiempo real
- Tracks activos con trayectorias visuales
- Líneas de conteo configurables
- Alertas de colisión potencial
- Captura de pantalla con un clic
- Dashboard informativo personalizable

### ⚡ Rendimiento y Optimización
- Procesamiento optimizado para CPU y GPU
- Pipeline asíncrono con múltiples workers
- Buffer circular preasignado para frames
- Pool de frames para reutilización de memoria
- Detección optimizada con ONNX Runtime
- Operaciones vectorizadas con Numba
- Gestión automática de memoria y GC
- Control de flujo adaptativo

### 🛠️ Desarrollo y Mantenimiento
- Código tipado con type hints
- Logging estructurado con contexto
- Validación centralizada de datos
- Manejo robusto de errores
- Sistema de configuración con Pydantic
- Pruebas de benchmark integradas

---

## 🏗️ Arquitectura

El sistema sigue una arquitectura de **pipeline modular** con componentes independientes y desacoplados:

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │ Capture  │───▶│   Buffer     │───▶│   Process    │           │
│  │ Thread   │    │  (Circular)  │    │    Pool      │           │
│  └──────────┘    └──────────────┘    └──────────────┘           │
│       │                │                  │                     │
│       ▼                ▼                  ▼                     │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                  Frame Buffer                       │        │
│  └─────────────────────────────────────────────────────┘        │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                  Render Thread                      │        │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │        │
│  │  │ Overlay  │  │ Dashboard│  │   System Info    │   │        │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Componentes Principales

| Componente | Descripción | Archivo Principal |
|------------|-------------|-------------------|
| **CaptureManager** | Gestión de captura de video con reconexión automática y control de flujo | `core/capture/manager.py` |
| **FrameBuffer** | Buffer circular optimizado con memoria preasignada | `core/pipeline/buffer.py` |
| **OptimizedThreadPool** | Pool de workers con priorización de tareas | `utils/thread_pool.py` |
| **YOLODetector** | Detector de objetos con soporte ONNX y caché | `core/detector/base.py` |
| **OptimizedYOLODetector** | Versión optimizada para CPU con ONNX Runtime y Numba | `core/detector/optimized.py` |
| **MultiObjectTracker** | Tracker con re-identificación, MHT y fusión de sensores | `core/tracker/base.py` |
| **VehicleCounter** | Contador de vehículos con análisis de trayectoria | `core/counter.py` |
| **FrameRenderer** | Renderizador por capas con caché de métricas | `core/pipeline/renderer.py` |

---

## 📦 Requisitos

### Sistema Operativo
- **Linux**: Ubuntu 20.04+, Debian 11+, Fedora 38+
- **Windows**: Windows 10/11 (64-bit)
- **macOS**: macOS 12+ (Apple Silicon y Intel)

### Hardware Recomendado

| Componente | Mínimo | Recomendado | Óptimo |
|------------|--------|-------------|--------|
| **RAM** | 4 GB | 8 GB | 16 GB |
| **CPU** | 4 núcleos | 8 núcleos | 8+ núcleos |
| **GPU** | — | NVIDIA GTX 1060+ (CUDA) | NVIDIA RTX 3060+ |
| **Almacenamiento** | 2 GB | 5 GB | 10 GB (SSD) |

### Dependencias Principales

```txt
# ── Core ──────────────────────────────────────
opencv-python>=4.8.0
ultralytics>=8.4.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
scikit-image>=0.20.0

# ── Configuración ─────────────────────────────
pydantic>=2.0.0
pyyaml>=6.0

# ── Optimización ──────────────────────────────
onnxruntime>=1.15.0
numba>=0.57.0
psutil>=5.9.0

# ── Machine Learning ──────────────────────────
torch>=2.0.0
torchvision>=0.15.0
```

---

## 🔧 Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/daencordova/traffic-insights.git
cd traffic-insights
```

### 2. Crear Entorno Virtual

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Descargar Modelo YOLO

El modelo se descargará **automáticamente** al ejecutar por primera vez. También puedes descargarlo manualmente:

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

---

## ⚙️ Configuración

### Archivo de Configuración (`config.yaml`)

```yaml
# ── Modelo de detección ───────────────────────
model:
  model_path: "yolov8n.pt"
  confidence_threshold: 0.35
  iou_threshold: 0.45
  vehicle_classes: [2, 3, 5, 7]  # Car, motorcycle, bus, truck
  imgsz: 320
  max_det: 10
  use_onnx: true

# ── Cámara ──────────────────────────────────────
camera:
  source: "0"  # 0 = cámara web, o ruta a video
  width: 640
  height: 480
  reconnect_attempts: 5
  reconnect_delay: 1.0

# ── Tracker avanzado ────────────────────────────
tracker:
  max_distance: 50.0
  max_frames_missed: 15
  min_hits_to_confirm: 2

  # Re-identificación
  enable_reidentification: true
  reid_similarity_threshold: 0.6

  # Predicción de trayectoria
  enable_path_prediction: true
  prediction_horizon: 2.0
  prediction_steps: 20

# ── Líneas de conteo ────────────────────────────
counting_lines:
  - id: "line_1"
    name: "Entrada"
    points: [[100, 300], [600, 300]]
    direction: "down"
    color: [0, 255, 0]

  - id: "line_2"
    name: "Salida"
    points: [[100, 400], [600, 400]]
    direction: "up"
    color: [0, 0, 255]

# ── Visualización ───────────────────────────────
visualization:
  show_trails: true
  show_dashboard: true
  show_fps: true
  trail_length: 30

# ── Optimización ────────────────────────────────
optimization:
  enable_batch_processing: true
  batch_size: 4
  max_workers: 4
  use_optimized_detector: true
  use_optimized_kalman: true
```

### Variables de Entorno (Opcional)

Puedes sobrescribir la configuración sin editar `config.yaml`:

```bash
export MODEL_PATH="yolov8n.pt"
export CAMERA_SOURCE="0"
export CONFIDENCE_THRESHOLD="0.4"
export USE_GPU="true"
```

---

## 🎮 Uso

### Ejecución Básica

```bash
# Modo asíncrono (recomendado para producción)
python main.py --config config.yaml

# Modo síncrono (legacy, útil para depuración)
python main.py --sync --config config.yaml

# Forzar modo CPU
python main.py --cpu-mode --config config.yaml
```

### Opciones de Línea de Comandos

| Opción | Abreviación | Descripción | Default |
|--------|-------------|-------------|---------|
| `--config` | `-c` | Ruta al archivo de configuración | `config.yaml` |
| `--source` | `-s` | Fuente de video (`0` para cámara) | Del config |
| `--sync` | — | Usar pipeline síncrono | `False` |
| `--workers` | `-w` | Número de workers | Auto-ajustado |
| `--buffer` | `-b` | Tamaño del buffer | Auto-ajustado |
| `--batch` | — | Habilitar procesamiento por lotes | Auto-ajustado |
| `--batch-size` | — | Tamaño del lote | `4` |
| `--cpu-mode` | — | Forzar modo CPU | `False` |
| `--verbose` | `-v` | Activar modo verbose | `False` |

### Ejemplos de Uso

```bash
# Usar cámara USB en modo CPU
python main.py --source 0 --cpu-mode --workers 2

# Procesar archivo de video con optimizaciones
python main.py --source video.mp4 --batch --batch-size 8

# Modo verbose para depuración
python main.py -v --config custom_config.yaml
```

---

## 🎮 Controles en Tiempo de Ejecución

| Tecla | Acción | Descripción |
|-------|--------|-------------|
| `Q` / `ESC` | **Salir** | Finaliza la ejecución del sistema |
| `SPACE` | **Pausar / Reanudar** | Pausa o reanuda el procesamiento |
| `S` | **Captura** | Guarda captura en `data/screenshots/` |
| `R` | **Reiniciar** | Reinicia contadores y tracker |
| `H` | **Ayuda** | Muestra controles en consola |
| `D` | **Diagnóstico** | Muestra información de diagnóstico |

---

## 📁 Estructura del Proyecto

```
traffic-insights/
├── config/
│   ├── __init__.py
│   ├── manager.py              # Gestor de configuración
│   └── settings.py             # Modelos Pydantic
│
├── core/
│   ├── __init__.py
│   ├── constants.py            # Constantes del sistema
│   ├── exceptions.py           # Excepciones personalizadas
│   ├── interfaces.py           # Interfaces del sistema
│   │
│   ├── validators/             # Validadores centralizados
│   │   ├── __init__.py
│   │   ├── frame_validator.py
│   │   ├── bbox_validator.py
│   │   └── detection_validator.py
│   │
│   ├── capture/                # Captura de video
│   │   ├── __init__.py
│   │   ├── manager.py          # CaptureManager
│   │   └── reconnector.py      # Reconexión automática
│   │
│   ├── detector/               # Detección de objetos
│   │   ├── __init__.py
│   │   ├── base.py             # YOLODetector
│   │   ├── optimized.py        # OptimizedYOLODetector
│   │   ├── cache.py            # DetectionCache
│   │   └── preprocessor.py     # ImagePreprocessor
│   │
│   ├── tracker/                # Tracking avanzado
│   │   ├── __init__.py
│   │   ├── base.py             # MultiObjectTracker
│   │   ├── matcher.py          # TrackMatcher
│   │   ├── reidentifier.py     # ReIDSystem
│   │   ├── mht_integration.py
│   │   ├── sensor_fusion.py
│   │   └── path_predictor.py
│   │
│   ├── pipeline/               # Pipelines de procesamiento
│   │   ├── __init__.py
│   │   ├── async_pipeline.py
│   │   ├── sync_pipeline.py
│   │   ├── renderer.py         # FrameRenderer
│   │   ├── dashboard.py        # DashboardRenderer
│   │   └── overlay.py          # OverlayRenderer
│   │
│   └── counter.py              # VehicleCounter
│
├── models/                     # Modelos de datos
│   ├── __init__.py
│   ├── enums.py
│   ├── track_state.py
│   ├── kalman.py
│   └── kalman_optimized.py
│
├── utils/                      # Utilidades
│   ├── __init__.py
│   ├── logger.py               # Logging estructurado
│   ├── helpers.py              # Funciones auxiliares
│   ├── geometry.py             # Operaciones geométricas
│   └── thread_pool.py          # Thread pool optimizado
│
├── data/                       # Datos generados
│   ├── screenshots/            # Capturas de pantalla
│   ├── exports/              # Exportaciones
│   └── logs/                 # Archivos de log
│
├── scripts/                    # Scripts auxiliares
│   ├── benchmark.py          # Benchmark de rendimiento
│   └── profile.py            # Perfilamiento
│
├── config.yaml                 # Configuración principal
├── requirements.txt            # Dependencias
├── main.py                     # Punto de entrada
├── run.sh                      # Script de ejecución
└── README.md                   # Este archivo
```

---

## 📖 Documentación Técnica

### Módulos Principales

#### 1. Detector (`core/detector/`)

**YOLODetector** — Detector base con YOLOv8:

```python
from core.detector import YOLODetector

detector = YOLODetector()
detections = detector.detect(frame)  # Lista de detecciones
```

**OptimizedYOLODetector** — Versión optimizada para CPU con ONNX Runtime y Numba:

```python
from core.detector import OptimizedYOLODetector

detector = OptimizedYOLODetector()
detections = detector.detect(frame)
```

#### 2. Tracker (`core/tracker/`)

**MultiObjectTracker** — Tracker híbrido con todas las características avanzadas:

```python
from core.tracker import MultiObjectTracker

tracker = MultiObjectTracker()
tracks = tracker.update(detections, frame)
```

#### 3. Pipeline (`core/pipeline/`)

**AsyncPipeline** — Pipeline asíncrono optimizado:

```python
from core.pipeline import AsyncPipeline

with AsyncPipeline() as pipeline:
    pipeline.start()
    while pipeline.is_running:
        time.sleep(0.1)
```

### Configuración por Código

```python
from config.manager import config_manager

# Cargar configuración
config = config_manager.load_from_file("config.yaml")

# Acceder a valores
confidence = config.model.confidence_threshold
source = config.camera.source

# Modificar en tiempo de ejecución
config_manager.set("model.confidence_threshold", 0.6)

# Obtener valor con default
value = config_manager.get("feature.disponible", default=False)
```

### Logging Estructurado

```python
from utils.logger import LoggerMixin

class MyComponent(LoggerMixin):
    def process(self):
        self.logger.info("Iniciando procesamiento", data={"items": 10})
        self.logger.warning("Memoria alta", memory_percent=85)
        self.logger.error("Error crítico", exc_info=True)
```

### Validación de Datos

```python
from core.validators import validate_frame, validate_detection

if validate_frame(frame):
    # Frame válido
    pass

result = validate_detection(detection)
if result.is_valid:
    # Detección válida
    pass
```

---

## 📊 Rendimiento

### Benchmark en CPU (Intel i7-9700K, 8 núcleos)

| Modo | FPS | Tiempo Promedio |
|------|-----|-----------------|
| Síncrono (PyTorch) | 8–12 | 83–125 ms |
| Síncrono (ONNX) | 15–20 | 50–67 ms |
| Asíncrono (4 workers) | 20–30 | 33–50 ms |
| Asíncrono + Batch | 25–35 | 28–40 ms |

### Modos de Ejecución

| Modo | CPU | GPU | Memoria | Uso Recomendado |
|------|-----|-----|---------|-----------------|
| **Síncrono** | ✅ | ✅ | Bajo | Depuración, pruebas |
| **Asíncrono** | ✅ | ✅ | Medio | Producción, tiempo real |
| **Asíncrono + Batch** | ✅ | ❌ | Medio | CPU, mejor rendimiento |
| **ONNX** | ✅ | ❌ | Bajo | CPU optimizado |

---

## 🔧 Solución de Problemas

### Errores Comunes

#### 1. "No se pudo abrir la fuente"

Verifica que la cámara esté disponible:

```bash
# Linux
ls /dev/video*
v4l2-ctl --list-devices

# Windows: usa Device Manager para verificar
```

Configura la fuente correcta en `config.yaml`:

```yaml
camera:
  source: "0"  # Cambiar según disponibilidad
```

#### 2. "Error de CUDA / Out of Memory"

```bash
# Forzar modo CPU
python main.py --cpu-mode
```

O en `config.yaml`:

```yaml
model:
  device: "cpu"
```

#### 3. "Bajo rendimiento / FPS bajos"

```yaml
# Configurar optimizaciones
optimization:
  enable_batch_processing: true
  batch_size: 4
  max_workers: 4
  use_optimized_detector: true

model:
  imgsz: 320        # Reducir tamaño de imagen
  use_onnx: true    # Activar ONNX
```

#### 4. "Error de dependencias"

```bash
# Reinstalar dependencias
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Logs del Sistema

Los logs se guardan en `data/logs/system.log`:

```text
2024-01-15 14:30:25 - main - INFO - 🚗 SISTEMA DE SEGUIMIENTO DE TRÁFICO v0.1.0
2024-01-15 14:30:25 - ConfigManager - INFO - 📄 Cargando configuración desde: config.yaml
2024-01-15 14:30:26 - YOLODetector - INFO - 🤖 DETECTOR YOLO inicializado
2024-01-15 14:30:27 - AsyncPipeline - INFO - Pipeline asíncrono iniciado
```

---

## 🤝 Contribuciones

### Guía para Contribuir

1. **Fork** el repositorio
2. Crea una rama para tu feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit tus cambios: `git commit -m "feat: agregar nueva funcionalidad"`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Crea un **Pull Request**

### Estilo de Código

| Aspecto | Convención |
|---------|------------|
| **Python** | PEP 8 con type hints |
| **Docstrings** | Formato Google |
| **Commits** | [Conventional Commits](https://www.conventionalcommits.org/) |

### Ejecutar Pruebas

```bash
# Benchmark de rendimiento
python scripts/benchmark.py

# Perfilamiento
python scripts/profile.py
```

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.

<div align="center">

---

Desarrollado con ❤️ por **Daniel Córdova**

⭐ Si te gusta este proyecto, ¡déjanos una estrella!

</div>
