from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import logging
from analysis_module import convert_to_wav, load_audio, analyze_pitch_with_librosa, analyze_timbre, analyze_dynamics, analyze_articulation, analyze_rhythm, analyze_breath_control, analyze_vibrato, plot_analysis_results, tune_audio
from supabase import create_client, Client
from io import BytesIO
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Supabase client
url: str = os.getenv('SUPABASE_URL')
key: str = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(url, key)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Read file content
        content = await file.read()
        
        # Upload original audio to Supabase directly to the bucket root
        audio_path = f"{file.filename}"
        supabase.storage.from_("audio").upload(audio_path, content)

        # Convert to WAV if necessary
        if not file.filename.endswith('.wav'):
            wav_content = convert_to_wav(BytesIO(content))
            audio_path = f"{file.filename.rsplit('.', 1)[0]}.wav"
            supabase.storage.from_("audio").upload(audio_path, wav_content.getvalue())
        else:
            wav_content = BytesIO(content)

        y, sr = load_audio(wav_content)
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

        logger.info(f"Analysis results: {results}")

        # Generate and upload graph
        graph_content = plot_analysis_results(results)
        graph_path = f"analysis_results_{file.filename}.png"
        supabase.storage.from_("graphs").upload(graph_path, graph_content.getvalue())

        # Generate and upload tuned audio
        tuned_content = tune_audio(wav_content, semitone_shift=2)
        tuned_path = f"tuned_{file.filename}"
        supabase.storage.from_("tuned_audio").upload(tuned_path, tuned_content.getvalue())

        # Generate signed URLs for files
        audio_url = supabase.storage.from_("audio").get_public_url(audio_path)
        graph_url = supabase.storage.from_("graphs").get_public_url(graph_path)
        tuned_url = supabase.storage.from_("tuned_audio").get_public_url(tuned_path)

        rhythm_tempo_value = rhythm_tempo if np.isscalar(rhythm_tempo) else rhythm_tempo[0]

        analysis_text = (
            f"Pitch Analysis: {np.mean(pitch_librosa):.2f} Hz\n"
            f"Timbre Analysis (Spectral Centroid): {np.mean(timbre):.2f}\n"
            f"Dynamics Analysis (RMS): {np.mean(dynamics):.2f}\n"
            f"Articulation Analysis (Onset Strength): {np.mean(articulation):.2f}\n"
            f"Rhythm Analysis (Tempo): {rhythm_tempo_value:.2f} BPM\n"
            f"Breath Control (RMS): {np.mean(breath_control):.2f}\n"
            f"Vibrato Analysis (Pitch): {np.mean(vibrato):.2f} Hz"
        )

        return JSONResponse(content={
            "audioUrl": audio_url,
            "graphUrl": graph_url,
            "tunedAudioUrl": tuned_url,
            "analysisText": analysis_text
        })
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
