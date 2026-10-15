"""
mllm_demo.py
============

Demo mínima y educativa de un Modelo de Lenguaje Multimodal (MLLM, por sus
siglas en inglés) usando los modelos de OpenAI con capacidad de visión
(por ejemplo, gpt-4o) más transcripción de audio con Whisper.

QUÉ HACE ESTE SCRIPT
---------------------
1. Escanea una carpeta "images" (jpg/png/webp/gif) y una carpeta "videos"
   (mp4/mov/avi/mkv), y muestra un menú numerado con todo lo disponible.
2. El usuario selecciona UN solo archivo (una imagen o un video) para
   trabajar con él. Así el modelo siempre sabe exactamente sobre qué se le
   está preguntando, sin mezclar contenido de varias fuentes:
   - Si es una imagen: se codifica tal cual.
   - Si es un video: se extraen unos cuantos fotogramas espaciados
     uniformemente con OpenCV (ya que la API de OpenAI no acepta archivos
     de video crudos, solo imágenes fijas) y además se extrae el audio con
     ffmpeg y se transcribe con Whisper, para poder responder también sobre
     lo que se dijo.
3. Se abre un ciclo de chat interactivo en consola donde las preguntas se
   responden únicamente en base al archivo seleccionado.
4. En cualquier momento se puede escribir "cambiar" para volver al menú y
   seleccionar otra imagen u otro video sin reiniciar el script.

USO
---
    export OPENAI_API_KEY="sk-..."
    python mllm_demo.py --images-dir ./images --videos-dir ./videos

    # Parámetros opcionales:
    #   --model gpt-4o                 (por defecto: gpt-4o)
    #   --frames-per-video 3           (por defecto: 3 fotogramas por video)
    #   --whisper-model whisper-1      (por defecto: whisper-1)

Comandos disponibles durante el chat:
    cambiar         -> vuelve al menú para seleccionar otro archivo
    exit / quit     -> termina el programa

REQUISITOS
----------
    pip install -r requirements.txt

    Además, se necesita tener "ffmpeg" instalado en el sistema (se usa para
    extraer el audio de los videos antes de transcribirlo). En macOS:
        brew install ffmpeg
    Si no está instalado, el script sigue funcionando pero se salta la
    transcripción de audio de los videos (solo usa fotogramas).

CONFIGURACIÓN
--------------
Coloque su llave de API en un archivo ".env" junto a este script:
    OPENAI_API_KEY=sk-...
"""

import argparse
import base64
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
COMANDOS_SALIR = {"exit", "quit", "salir"}
COMANDOS_CAMBIAR = {"cambiar", "seleccionar", "menu"}

FFMPEG_DISPONIBLE = shutil.which("ffmpeg") is not None


@dataclass
class Seleccion:
    """Contenido ya cargado y listo para enviarle al modelo: una sola imagen o un solo video."""

    nombre: str
    tipo: str  # "imagen" o "video"
    image_urls: list[str]
    transcript: str  # transcripción del audio (vacía si es imagen o no tiene audio)


def parse_args() -> argparse.Namespace:
    """Lee las carpetas a escanear y la configuración del modelo desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Hágale preguntas a un MLLM de OpenAI sobre una imagen o un video seleccionado."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("images"),
        help="Carpeta con las imágenes disponibles para seleccionar (por defecto: ./images)",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("videos"),
        help="Carpeta con los videos disponibles para seleccionar (por defecto: ./videos)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Modelo de OpenAI con capacidad de visión a usar (por defecto: gpt-4o)",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=3,
        help="Cantidad de fotogramas espaciados uniformemente a extraer por video (por defecto: 3)",
    )
    parser.add_argument(
        "--whisper-model",
        default="whisper-1",
        help="Modelo de OpenAI para transcribir el audio de los videos (por defecto: whisper-1)",
    )
    return parser.parse_args()


def encode_image_bytes(image_bytes: bytes) -> str:
    """Convierte bytes crudos de una imagen en un data URL base64 que acepta la API de OpenAI."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def list_available_files(images_dir: Path, videos_dir: Path) -> list[Path]:
    """Devuelve la lista combinada de imágenes y videos disponibles para seleccionar."""
    files = []

    if images_dir.is_dir():
        files.extend(
            p
            for p in sorted(images_dir.iterdir())
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        print(f"[aviso] no se encontró la carpeta de imágenes: {images_dir}")

    if videos_dir.is_dir():
        files.extend(
            p
            for p in sorted(videos_dir.iterdir())
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )
    else:
        print(f"[aviso] no se encontró la carpeta de videos: {videos_dir}")

    return files


def print_menu(files: list[Path]) -> None:
    """Imprime la lista numerada de imágenes y videos disponibles para seleccionar."""
    print("\nArchivos disponibles:")
    for i, path in enumerate(files, start=1):
        tipo = "imagen" if path.suffix.lower() in IMAGE_EXTENSIONS else "video"
        print(f"  [{i}] ({tipo}) {path.name}")


def prompt_selection(files: list[Path]) -> Path | None:
    """Pide al usuario el número del archivo a seleccionar. Devuelve None si cancela."""
    while True:
        choice = (
            input("\nSeleccione un número (o 'salir' para cancelar): ").strip().lower()
        )
        if choice in COMANDOS_SALIR:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print("Opción inválida, intente de nuevo.")


def extract_frames_from_video(path: Path, num_frames: int) -> list[str]:
    """
    Extrae `num_frames` fotogramas espaciados uniformemente de un video y los
    devuelve como data URLs en base64. Así es como logramos que un MLLM que
    solo entiende imágenes pueda "ver" un video: lo convertimos en una
    secuencia corta de imágenes fijas.
    """
    data_urls = []
    capture = cv2.VideoCapture(str(path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"[aviso] no se pudieron leer fotogramas de {path.name}")
        capture.release()
        return data_urls

    # Se espacian uniformemente los índices de los fotogramas a muestrear en todo el video.
    step = max(total_frames // num_frames, 1)
    target_indices = [i * step for i in range(num_frames) if i * step < total_frames]

    for index in target_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = capture.read()
        if not success:
            continue
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue
        data_urls.append(encode_image_bytes(buffer.tobytes()))

    capture.release()
    print(f"[fotogramas extraídos] {path.name} -> {len(data_urls)} fotograma(s)")
    return data_urls


def transcribe_video_audio(client: OpenAI, path: Path, whisper_model: str) -> str:
    """
    Extrae la pista de audio de un video con ffmpeg y la transcribe a texto
    usando el modelo Whisper de OpenAI. Devuelve una cadena vacía si el
    video no tiene audio, si ffmpeg no está instalado, o si algo falla.
    """
    if not FFMPEG_DISPONIBLE:
        print("[aviso] ffmpeg no está instalado: se omite la transcripción de audio")
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = Path(tmp.name)

    try:
        # -vn: descarta el video, nos quedamos solo con el audio.
        # 16 kHz mono en WAV es un formato liviano y suficiente para Whisper.
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(audio_path),
            ],
            capture_output=True,
        )

        if result.returncode != 0 or audio_path.stat().st_size == 0:
            print(f"[aviso] {path.name} no tiene audio o no se pudo extraer")
            return ""

        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=whisper_model,
                file=audio_file,
            )

        text = transcription.text.strip()
        if text:
            print(f"[audio transcrito] {path.name} -> {len(text)} caracteres")
        return text
    finally:
        audio_path.unlink(missing_ok=True)


def load_selection(
    client: OpenAI, path: Path, frames_per_video: int, whisper_model: str
) -> Seleccion:
    """
    Carga el archivo que el usuario seleccionó del menú:
    - Si es una imagen, se codifica directamente.
    - Si es un video, se extraen fotogramas y se transcribe el audio.
    """
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        image_bytes = path.read_bytes()
        return Seleccion(
            nombre=path.name,
            tipo="imagen",
            image_urls=[encode_image_bytes(image_bytes)],
            transcript="",
        )

    frame_urls = extract_frames_from_video(path, frames_per_video)
    transcript = transcribe_video_audio(client, path, whisper_model)
    return Seleccion(
        nombre=path.name, tipo="video", image_urls=frame_urls, transcript=transcript
    )


def ask_model(client: OpenAI, model: str, question: str, seleccion: Seleccion) -> str:
    """Envía la pregunta al modelo de OpenAI junto con el contenido de la selección actual."""
    text_parts = [f'Está viendo el siguiente {seleccion.tipo}: "{seleccion.nombre}".']
    if seleccion.transcript:
        text_parts.append("Transcripción del audio del video:\n" + seleccion.transcript)
    text_parts.append(f"Pregunta: {question}")

    content = [{"type": "text", "text": "\n\n".join(text_parts)}]
    for url in seleccion.image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


def elegir_y_cargar(
    client: OpenAI, files: list[Path], frames_per_video: int, whisper_model: str
) -> Seleccion | None:
    """Muestra el menú, pide una selección y carga el archivo elegido. Devuelve None si se cancela."""
    print_menu(files)
    selected_path = prompt_selection(files)
    if selected_path is None:
        return None

    print(f"\nCargando {selected_path.name}...")
    seleccion = load_selection(client, selected_path, frames_per_video, whisper_model)
    print(f'¡Listo! Ahora puede preguntar sobre "{seleccion.nombre}".')
    print(
        f"(Escriba '{next(iter(COMANDOS_CAMBIAR))}' para elegir otro archivo, 'exit' para salir.)\n"
    )
    return seleccion


def main() -> None:
    load_dotenv()  # lee OPENAI_API_KEY (y cualquier otra variable) desde un archivo .env local
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "Error: defina OPENAI_API_KEY en su archivo .env o en el entorno primero."
        )

    client = OpenAI()

    files = list_available_files(args.images_dir, args.videos_dir)
    if not files:
        sys.exit(
            "No se encontraron imágenes ni videos disponibles. Agregue archivos a las carpetas y vuelva a intentar."
        )

    seleccion = elegir_y_cargar(
        client, files, args.frames_per_video, args.whisper_model
    )
    if seleccion is None:
        return

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        low = question.lower()

        if low in COMANDOS_SALIR:
            break

        if low in COMANDOS_CAMBIAR:
            nueva_seleccion = elegir_y_cargar(
                client, files, args.frames_per_video, args.whisper_model
            )
            if nueva_seleccion is not None:
                seleccion = nueva_seleccion
            continue

        if not question:
            continue

        answer = ask_model(client, args.model, question, seleccion)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
