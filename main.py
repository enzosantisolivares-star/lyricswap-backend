import os
import uuid
import shutil
import tempfile
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import anthropic
import requests
import whisper

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
        # 1. Guardar audio
        suffix = Path(audio.filename).suffix or ".mp3"
        original_path = session_dir / f"original{suffix}"
        content = await audio.read()
        with open(original_path, "wb") as f:
            f.write(content)

        # 2. Demucs
        demucs_output = session_dir / "demucs"
        demucs_output.mkdir(exist_ok=True)
        proc = subprocess.run(
            ["python", "-m", "demucs", "--two-stems=vocals", "-o", str(demucs_output), str(original_path)],
            capture_output=True, text=True, timeout=900
        )
        vocals_path = next(demucs_output.rglob("vocals.wav"), None)
        no_vocals_path = next(demucs_output.rglob("no_vocals.wav"), None)
        if not vocals_path or not no_vocals_path:
            files = [str(p) for p in demucs_output.rglob("*")]
            raise HTTPException(status_code=500, detail=f"Demucs no genero archivos. stderr: {proc.stderr[-200:]} | files: {files[:5]}")

        # 3. Whisper
        model = get_whisper_model()
        transcription = model.transcribe(str(vocals_path), language=language)
        original_lyrics = transcription["text"].strip()
        if not original_lyrics:
            raise HTTPException(status_code=400, detail="No se pudo transcribir")

        # 4. Claude
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-opus-4-5", max_tokens=2048,
            messages=[{"role": "user", "content": f"Reescribe esta letra manteniendo el ritmo y metrica exacta.\nLETRA ORIGINAL:\n{original_lyrics}\nNUEVO TEMA:\n{theme}\nResponde SOLO la nueva letra."}]
        )
        new_lyrics = message.content[0].text.strip()

        # 5. ElevenLabs
        new_vocals_path = session_dir / "new_vocals.mp3"
        el_resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={"text": new_lyrics, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}},
            timeout=120
        )
        if el_resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"ElevenLabs error: {el_resp.text[:200]}")
        with open(new_vocals_path, "wb") as f:
            f.write(el_resp.content)

        # 6. FFmpeg mix
        output_path = session_dir / "output.mp3"
        ffmpeg = subprocess.run(
            ["ffmpeg", "-y", "-i", str(no_vocals_path), "-i", str(new_vocals_path),
             "-filter_complex", "[0:a]volume=1.0[inst];[1:a]volume=1.2[voc];[inst][voc]amix=inputs=2:duration=longest:dropout_transition=2[out]",
             "-map", "[out]", "-ar", "44100", "-ab", "192k", str(output_path)],
            capture_output=True, text=True, timeout=120
        )
        if ffmpeg.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {ffmpeg.stderr[-200:]}")

        return FileResponse(
            path=str(output_path), media_type="audio/mpeg", filename="lyricswap_output.mp3",
            headers={
                "X-Original-Lyrics": original_lyrics[:500],
                "X-New-Lyrics": new_lyrics[:500],
                "Access-Control-Expose-Headers": "X-Original-Lyrics,X-New-Lyrics"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
def list_voices():
    resp = requests.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": ELEVENLABS_API_KEY})
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Error obteniendo voces")
    return [{"voice_id": v["voice_id"], "name": v["name"]} for v in resp.json().get("voices", [])]

frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
