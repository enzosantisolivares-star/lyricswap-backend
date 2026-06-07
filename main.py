import os
import uuid
import shutil
import tempfile
import subprocess
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import anthropic
import requests
import whisper
import json

app = FastAPI(title="LyricSwap API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TEMP_DIR = Path(tempfile.gettempdir()) / "lyricswap"
TEMP_DIR.mkdir(exist_ok=True)
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("base")
    return whisper_model

def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

@app.get("/health")
def health():
    return {"status": "ok", "service": "LyricSwap API"}

@app.post("/swap")
async def swap_lyrics(
    audio: UploadFile = File(...),
    theme: str = Form(...),
    voice_id: str = Form("21m00Tcm4TlvDq8ikWAM"),
    language: str = Form("es"),
):
    session_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    try:
        original_path = session_dir / f"original{Path(audio_filename).suffix}"
        with open(original_path, "wb") as f:
            f.write(audio_bytes)

        demucs_output = session_dir / "demucs"
        demucs_output.mkdir(exist_ok=True)
        result = subprocess.run(
            ["python", "-m", "demucs", "--two-stems=vocals", "-o", str(demucs_output), str(original_path)],
            capture_output=True, text=True, timeout=900
        )
        vocals_path = next(demucs_output.rglob("vocals.wav"), None)
        no_vocals_path = next(demucs_output.rglob("no_vocals.wav"), None)
        if not vocals_path or not no_vocals_path:
            all_files = [str(p) for p in demucs_output.rglob("*")]
            raise HTTPException(status_code=500, detail=f"Demucs no generÃ³ archivos. stderr: {result.stderr[-300:]} files: {all_files[:5]}")

        model = get_whisper_model()
        transcription = model.transcribe(str(vocals_path), language=language)
        original_lyrics = transcription["text"].strip()
        if not original_lyrics:
            raise HTTPException(status_code=400, detail="No se pudo transcribir")

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = f"Reescribe esta letra manteniendo el ritmo y mÃ©trica exacta.\nLETRA ORIGINAL:\n{original_lyrics}\nNUEVO TEMA:\n{theme}\nResponde SOLO la nueva letra."
        message = client.messages.create(model="claude-opus-4-5", max_tokens=2048, messages=[{"role": "user", "content": prompt}])
        new_lyrics = message.content[0].text.strip()

        new_vocals_path = session_dir / "new_vocals.mp3"
        el_response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={"text": new_lyrics, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}},
            timeout=120
        )
        if el_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"ElevenLabs error: {el_response.text}")
        with open(new_vocals_path, "wb") as f:
            f.write(el_response.content)

        output_path = session_dir / "lyricswap_output.mp3"
        ffmpeg_result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(no_vocals_path), "-i", str(new_vocals_path),
             "-filter_complex", "[0:a]volume=1.0[inst];[1:a]volume=1.2[voc];[inst][voc]amix=inputs=2:duration=longest:dropout_transition=2[out]",
             "-map", "[out]", "-ar", "44100", "-ab", "192k", str(output_path)],
            capture_output=True, text=True, timeout=120
        )
        if ffmpeg_result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {ffmpeg_result.stderr}")

        return FileResponse(
            path=str(output_path), media_type="audio/mpeg",
            filename="lyricswap_output.mp3",
            headers={
                "X-Original-Lyrics": original_lyrics[:500],
                "X-New-Lyrics": new_lyrics[:500],
                "Access-Control-Expose-Headers": "X-Original-Lyrics, X-New-Lyrics"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pass

@app.post("/swap-stream")
async def swap_lyrics_stream(
    audio: UploadFile = File(...),
    theme: str = Form(...),
    voice_id: str = Form("21m00Tcm4TlvDq8ikWAM"),
    language: str = Form("es"),
):
    audio_bytes = await audio.read()
    audio_filename = audio.filename
    async def generate():
        session_id = str(uuid.uuid4())
        session_dir = TEMP_DIR / session_id
        session_dir.mkdir(exist_ok=True)
        try:
            yield sse("progress", {"step": "upload", "pct": 5, "msg": "Archivo recibido, separando stems..."})
            original_path = session_dir / f"original{Path(audio_filename).suffix}"
            with open(original_path, "wb") as f:
                f.write(audio_bytes)

            yield sse("progress", {"step": "demucs", "pct": 15, "msg": "Demucs separando voz del beat..."})
            demucs_output = session_dir / "demucs"
            demucs_output.mkdir(exist_ok=True)
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "demucs", "--two-stems=vocals", "-o", str(demucs_output), str(original_path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            pct = 15
            while True:
                line = await asyncio.wait_for(proc.stderr.readline(), timeout=600)
                if not line:
                    break
                txt = line.decode(errors="ignore").strip()
                if "%" in txt:
                    try:
                        p = int(txt.split("%")[0].split("|")[-1].strip())
                        pct = 15 + int(p * 0.35)
                        yield sse("progress", {"step": "demucs", "pct": pct, "msg": f"Demucs: {p}% completado..."})
                    except:
                        pass
            await proc.wait()

            vocals_path = next(demucs_output.rglob("vocals.wav"), None)
            no_vocals_path = next(demucs_output.rglob("no_vocals.wav"), None)
            if not vocals_path or not no_vocals_path:
                all_files = [str(p) for p in demucs_output.rglob("*")]
                yield sse("error", {"msg": f"Demucs no generÃ³ archivos. Files: {all_files[:5]}"})
                return

            yield sse("progress", {"step": "whisper", "pct": 55, "msg": "Whisper transcribiendo la voz..."})
            model = get_whisper_model()
            transcription = model.transcribe(str(vocals_path), language=language)
            original_lyrics = transcription["text"].strip()
            if not original_lyrics:
                yield sse("error", {"msg": "No se pudo transcribir el audio"})
                return

            yield sse("progress", {"step": "claude", "pct": 65, "msg": "Claude reescribiendo la letra..."})
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            prompt = f"Reescribe esta letra manteniendo el ritmo y mÃ©trica exacta.\nLETRA ORIGINAL:\n{original_lyrics}\nNUEVO TEMA:\n{theme}\nResponde SOLO la nueva letra."
            message = client.messages.create(model="claude-opus-4-5", max_tokens=2048, messages=[{"role": "user", "content": prompt}])
            new_lyrics = message.content[0].text.strip()

            yield sse("progress", {"step": "elevenlabs", "pct": 78, "msg": "ElevenLabs sintetizando nueva voz..."})
            new_vocals_path = session_dir / "new_vocals.mp3"
            el_response = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
                json={"text": new_lyrics, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}},
                timeout=120
            )
            if el_response.status_code != 200:
                yield sse("error", {"msg": f"ElevenLabs error: {el_response.text[:200]}"})
                return
            with open(new_vocals_path, "wb") as f:
                f.write(el_response.content)

            yield sse("progress", {"step": "mix", "pct": 90, "msg": "FFmpeg mezclando voz + beat..."})
            output_path = session_dir / "lyricswap_output.mp3"
            ffmpeg_result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(no_vocals_path), "-i", str(new_vocals_path),
                 "-filter_complex", "[0:a]volume=1.0[inst];[1:a]volume=1.2[voc];[inst][voc]amix=inputs=2:duration=longest:dropout_transition=2[out]",
                 "-map", "[out]", "-ar", "44100", "-ab", "192k", str(output_path)],
                capture_output=True, text=True, timeout=120
            )
            if ffmpeg_result.returncode != 0:
                yield sse("error", {"msg": f"FFmpeg error: {ffmpeg_result.stderr[-200:]}"})
                return

            yield sse("done", {
                "pct": 100,
                "msg": "Â¡CanciÃ³n lista!",
                "session_id": session_id,
                "original_lyrics": original_lyrics[:500],
                "new_lyrics": new_lyrics[:500]
            })
        except asyncio.TimeoutError:
            yield sse("error", {"msg": "Timeout â el audio es muy largo, intenta con uno mÃ¡s corto"})
        except Exception as e:
            yield sse("error", {"msg": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/download/{session_id}")
async def download(session_id: str):
    output_path = TEMP_DIR / session_id / "lyricswap_output.mp3"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(path=str(output_path), media_type="audio/mpeg", filename="lyricswap_output.mp3")

@app.post("/transcribe")
async def transcribe_only(audio: UploadFile = File(...), language: str = Form("es")):
    session_dir = TEMP_DIR / str(uuid.uuid4())
    session_dir.mkdir(exist_ok=True)
    try:
        audio_path = session_dir / audio.filename
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        model = get_whisper_model()
        result = model.transcribe(str(audio_path), language=language)
        return {"lyrics": result["text"].strip(), "segments": result.get("segments", [])}
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)

@app.get("/voices")
def list_voices():
    resp = requests.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": ELEVENLABS_API_KEY})
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Error obteniendo voces")
    return [{"voice_id": v["voice_id"], "name": v["name"]} for v in resp.json().get("voices", [])]

# Servir frontend al final
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
