from fastapi import logger
import librosa
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from pydub import AudioSegment
import ffmpeg
import soundfile as sf

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

def tune_audio(file_content: BytesIO, semitone_shift: int) -> BytesIO:
    sound = AudioSegment.from_wav(file_content)
    tuned_sound = sound._spawn(sound.raw_data, overrides={"frame_rate": int(sound.frame_rate * (2 ** (semitone_shift / 12.0)))})
    tuned_content = BytesIO()
    tuned_sound.export(tuned_content, format='wav')
    tuned_content.seek(0)
    return tuned_content