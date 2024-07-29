from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import logging
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from io import BytesIO
from supabase import create_client, Client
import os
import ffmpeg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
url: str = os.getenv('SUPABASE_URL')
key: str = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(url, key)

def convert_to_wav(file_content: BytesIO, original_format: str) -> BytesIO:
    output = BytesIO()
    input_stream = ffmpeg.input('pipe:0', format=original_format)
    output_stream = ffmpeg.output(input_stream, 'pipe:1', format='wav', acodec='pcm_s16le')
    out, err = ffmpeg.run(output_stream, input=file_content.getvalue(), capture_stdout=True, capture_stderr=True)
    output.write(out)
    output.seek(0)
    return output

def load_audio(file_content: BytesIO):
    try:
        # First, try to load the audio directly
        y, sr = librosa.load(file_content, sr=None)
    except Exception as e:
        logger.warning(f"Could not load audio directly: {e}")
        try:
            # If direct loading fails, try to use soundfile to read the audio
            file_content.seek(0)
            data, samplerate = sf.read(file_content)
            y = data.T if data.ndim > 1 else data
            sr = samplerate
        except Exception as e:
            logger.error(f"Failed to load audio using soundfile: {e}")
            raise ValueError("Could not load the audio file. The format may be unsupported.")
    
    return y, sr

def analyze_pitch_with_librosa(y, sr):
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    return np.mean(pitches, axis=0)

def analyze_timbre(y, sr):
    return librosa.feature.spectral_centroid(y=y, sr=sr)[0]

def analyze_dynamics(y):
    return librosa.feature.rms(y=y)[0]

def analyze_articulation(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    return onset_env

def analyze_rhythm(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return tempo, beats

def analyze_breath_control(y, sr):
    return librosa.feature.rms(y=y)[0]

def analyze_vibrato(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return f0

def plot_analysis_results(results):
    num_plots = len(results)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots))
    plt.subplots_adjust(hspace=0.8)
    
    for i, (title, data) in enumerate(results.items()):
        if isinstance(data, list) and len(data) > 0:
            axs[i].plot(data)
        elif isinstance(data, np.ndarray) and data.size > 0:
            axs[i].plot(data)
        else:
            axs[i].text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
        axs[i].set_title(title, fontsize=8, loc='left', x=0.51)
        axs[i].tick_params(axis='both', which='major', labelsize=6)
        axs[i].title.set_fontsize(10)
        axs[i].title.set_size(8)
        axs[i].xaxis.label.set_size(6)
        axs[i].yaxis.label.set_size(6)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        content = await file.read()
        # logger.info(f"Received file: {file.filename}, size: {len(content)} bytes")
        
        # Determine the file format
        file_format = file.filename.split('.')[-1].lower()
        # logger.info(f"File format: {file_format}")
        
        # Convert to WAV if necessary
        if file_format != 'wav':
            wav_content = convert_to_wav(BytesIO(content), file_format)
        else:
            wav_content = BytesIO(content)

        # Upload original audio to Supabase
        audio_path = f"{file.filename}"
        upload_result = supabase.storage.from_("audio").upload(audio_path, content)
        # logger.info(f"Supabase upload result: {upload_result}")

        try:
            y, sr = load_audio(wav_content)
            # logger.info(f"Audio loaded: sample rate = {sr}, length = {len(y)}")
        except ValueError as e:
            logger.error(f"Failed to load audio: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        # Perform analyses
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

        # Generate signed URLs for files
        audio_url = supabase.storage.from_("audio").get_public_url(audio_path)
        graph_url = supabase.storage.from_("graphs").get_public_url(graph_path)

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
            "analysisText": analysis_text
        })
    except Exception as e:
        logger.error(f"Error in analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)