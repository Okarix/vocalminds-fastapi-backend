from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import logging
import matplotlib.pyplot as plt
from io import BytesIO
from supabase import create_client, Client
import os

from analysis_module import (
    convert_to_wav, load_audio, analyze_pitch_with_librosa, analyze_timbre,
    analyze_dynamics, analyze_articulation, analyze_rhythm, analyze_breath_control,
    analyze_vibrato, plot_analysis_results, tune_audio
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

url: str = os.getenv('SUPABASE_URL')
key: str = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(url, key)

@app.post("/analyze")
async def analyze_and_tune(file: UploadFile = File(...), semitone_shift: int = 0):
    try:
        content = await file.read()
        
        file_format = file.filename.split('.')[-1].lower()
        
        if file_format != 'wav':
            wav_content = convert_to_wav(BytesIO(content), file_format)
        else:
            wav_content = BytesIO(content)

        audio_path = f"{file.filename}"
        upload_result = supabase.storage.from_("audio").upload(audio_path, content)

        try:
            y, sr = load_audio(wav_content)
        except ValueError as e:
            logger.error(f"Failed to load audio: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        pitch_librosa = analyze_pitch_with_librosa(y, sr)
        timbre = analyze_timbre(y, sr)
        dynamics = analyze_dynamics(y)
        articulation = analyze_articulation(y, sr)
        rhythm_tempo, rhythm_beats = analyze_rhythm(y, sr)
        breath_control = analyze_breath_control(y, sr)
        vibrato = analyze_vibrato(y, sr)

        results = {
            "Pitch (Librosa)": pitch_librosa,
            "Timbre (Spectral Centroid)": timbre,
            "Dynamics (RMS)": dynamics,
            "Articulation (Onset Strength)": articulation,
            "Rhythm (Tempo)": [rhythm_tempo] * len(rhythm_beats),
            "Breath Control (RMS)": breath_control,
            "Vibrato (Pitch)": vibrato
        }

        graph_content = plot_analysis_results(results)
        graph_path = f"analysis_results_{file.filename}.png"
        supabase.storage.from_("graphs").upload(graph_path, graph_content.getvalue())

        audioUrl = supabase.storage.from_("audio").get_public_url(audio_path)
        graphUrl = supabase.storage.from_("graphs").get_public_url(graph_path)

        rhythm_tempo_value = rhythm_tempo if np.isscalar(rhythm_tempo) else rhythm_tempo[0]

        analysisText = (
            f"Pitch Analysis: {np.mean(pitch_librosa):.2f} Hz\n"
            f"Timbre Analysis (Spectral Centroid): {np.mean(timbre):.2f}\n"
            f"Dynamics Analysis (RMS): {np.mean(dynamics):.2f}\n"
            f"Articulation Analysis (Onset Strength): {np.mean(articulation):.2f}\n"
            f"Rhythm Analysis (Tempo): {rhythm_tempo_value:.2f} BPM\n"
            f"Breath Control (RMS): {np.mean(breath_control):.2f}\n"
            f"Vibrato Analysis (Pitch): {np.mean(vibrato):.2f} Hz"
        )

        wav_content.seek(0)
        tuned_content = tune_audio(wav_content, semitone_shift)
        
        tuned_audio_path = f"tuned_{file.filename}"
        supabase.storage.from_("tuned_audio").upload(tuned_audio_path, tuned_content.getvalue())
        
        tunedAudioUrl = supabase.storage.from_("tuned_audio").get_public_url(tuned_audio_path)

        return JSONResponse(content={
            "originalAudioUrl": audioUrl,
            "tunedAudioUrl": tunedAudioUrl,
            "graphUrl": graphUrl,
            "analysisText": analysisText
        })

    except Exception as e:
        logger.error(f"Error in analysis and tuning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)