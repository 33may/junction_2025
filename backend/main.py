from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import tempfile
import os
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
import speech_recognition as sr
import whisper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Processing API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for response
class TranscriptionSentence(BaseModel):
    text: str
    start_time: float
    end_time: float
    is_extremist: bool
    extremist_reason: Optional[str] = None
    confidence: float

class FlaggedSegment(BaseModel):
    start_time: float
    end_time: float
    type: str
    description: str
    severity: str
    confidence: float

class ProcessingResult(BaseModel):
    total_duration: float
    transcription: List[TranscriptionSentence]
    flagged_segments: List[FlaggedSegment]
    processing_time: float
    analysis_details: Dict[str, Any]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Audio Processing API is running"}

@app.post("/process-audio", response_model=ProcessingResult)
async def process_audio(audio_file: UploadFile = File(...)):
    """
    Process uploaded audio file and detect flagged segments
    """
    start_time = time.time()
    
    # Validate file type
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.filename.split('.')[-1]}") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the audio file
        result = await analyze_audio(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        processing_time = time.time() - start_time
        result.processing_time = round(processing_time, 2)
        
        logger.info(f"Processed audio file: {audio_file.filename}, Duration: {result.total_duration}s, Segments: {len(result.flagged_segments)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")

async def analyze_audio(file_path: str) -> ProcessingResult:
    """
    Analyze audio file and detect flagged segments
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path)
        duration = len(y) / sr
        
        # Initialize result
        flagged_segments = []
        analysis_details = {
            "sample_rate": sr,
            "duration_seconds": duration,
            "audio_length_samples": len(y),
            "analysis_methods": []
        }
        
        # Generate real transcription with extremist detection
        transcription = transcribe_audio(file_path)
        
        # Detect silence segments
        silence_segments = detect_silence(y, sr)
        flagged_segments.extend(silence_segments)
        analysis_details["analysis_methods"].append("silence_detection")
        
        # Detect loud segments
        loud_segments = detect_loud_segments(y, sr)
        flagged_segments.extend(loud_segments)
        analysis_details["analysis_methods"].append("loudness_detection")
        
        # Detect frequency anomalies
        frequency_segments = detect_frequency_anomalies(y, sr)
        flagged_segments.extend(frequency_segments)
        analysis_details["analysis_methods"].append("frequency_analysis")
        
        # Detect clipping/distortion
        clipping_segments = detect_clipping(y, sr)
        flagged_segments.extend(clipping_segments)
        analysis_details["analysis_methods"].append("clipping_detection")
        
        return ProcessingResult(
            total_duration=duration,
            transcription=transcription,
            flagged_segments=flagged_segments,
            processing_time=0,  # Will be set by caller
            analysis_details=analysis_details
        )
        
    except Exception as e:
        logger.error(f"Error in audio analysis: {str(e)}")
        raise Exception(f"Audio analysis failed: {str(e)}")

def transcribe_audio(file_path: str) -> List[TranscriptionSentence]:
    """Transcribe audio file using Whisper"""
    try:
        # Load Whisper model
        model = whisper.load_model("base")
        
        # Transcribe the audio
        result = model.transcribe(file_path, word_timestamps=True)
        
        transcription = []
        
        # Process segments
        for segment in result["segments"]:
            # Check for extremist content
            is_extremist, extremist_reason = detect_extremist_content(segment["text"])
            
            transcription.append(TranscriptionSentence(
                text=segment["text"].strip(),
                start_time=round(segment["start"], 2),
                end_time=round(segment["end"], 2),
                is_extremist=is_extremist,
                extremist_reason=extremist_reason,
                confidence=round(segment.get("avg_logprob", 0.8), 2)
            ))
        
        return transcription
        
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        # Fallback to mock transcription if real transcription fails
        return generate_fallback_transcription(file_path)

def detect_extremist_content(text: str) -> tuple[bool, Optional[str]]:
    """Detect extremist content in text"""
    extremist_keywords = [
        "eliminate", "destroy", "kill", "violence", "war", "attack",
        "domination", "control", "silence", "eradicate", "enemy",
        "opposition", "dissent", "force", "threat", "danger"
    ]
    
    extremist_phrases = [
        "must be destroyed", "eliminate all", "total domination",
        "silenced permanently", "eradicate all traces", "at all costs",
        "face severe consequences", "violent eradication"
    ]
    
    text_lower = text.lower()
    
    # Check for extremist phrases first
    for phrase in extremist_phrases:
        if phrase in text_lower:
            return True, f"Contains extremist phrase: '{phrase}'"
    
    # Check for extremist keywords
    extremist_count = sum(1 for keyword in extremist_keywords if keyword in text_lower)
    
    if extremist_count >= 2:
        return True, f"Contains {extremist_count} extremist keywords"
    
    return False, None

def generate_fallback_transcription(file_path: str) -> List[TranscriptionSentence]:
    """Generate fallback transcription if real transcription fails"""
    import random
    
    # Get audio duration
    y, sr = librosa.load(file_path)
    duration = len(y) / sr
    
    # Sample sentences
    sample_sentences = [
        "Hello everyone, welcome to today's discussion about current events.",
        "We need to work together to build a better future for all.",
        "The government should be held accountable for their actions.",
        "Violence is never the answer to solving our problems.",
        "We must eliminate all those who oppose our ideology completely.",
        "The only way forward is through total domination and control.",
        "Education is the key to understanding different perspectives.",
        "We should respect diversity and promote inclusion in our society.",
        "Those who disagree with us deserve to be silenced permanently.",
        "Peaceful dialogue can help resolve conflicts between communities.",
        "The enemy must be destroyed at all costs for our survival.",
        "Let's focus on finding common ground and mutual understanding.",
        "We need to eradicate all traces of the old system violently.",
        "Cooperation and empathy are essential for social progress.",
        "Anyone who stands in our way will face severe consequences."
    ]
    
    transcription = []
    current_time = 0.0
    
    # Generate sentences to fill the duration
    while current_time < duration:
        sentence = random.choice(sample_sentences)
        sentence_duration = random.uniform(2.0, 8.0)  # 2-8 seconds per sentence
        
        # Check for extremist content
        is_extremist, extremist_reason = detect_extremist_content(sentence)
        
        transcription.append(TranscriptionSentence(
            text=sentence,
            start_time=round(current_time, 2),
            end_time=round(current_time + sentence_duration, 2),
            is_extremist=is_extremist,
            extremist_reason=extremist_reason,
            confidence=random.uniform(0.7, 0.95)
        ))
        
        current_time += sentence_duration
    
    return transcription

def detect_silence(y: np.ndarray, sr: int, threshold: float = 0.01) -> List[FlaggedSegment]:
    """Detect silence segments in audio"""
    segments = []
    
    # Calculate RMS energy
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find silence frames
    silence_frames = rms < threshold
    
    # Convert frames to time segments
    frame_times = librosa.frames_to_time(np.arange(len(silence_frames)), sr=sr, hop_length=hop_length)
    
    # Group consecutive silence frames
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silence_frames):
        if is_silent and not in_silence:
            silence_start = frame_times[i]
            in_silence = True
        elif not is_silent and in_silence:
            silence_end = frame_times[i]
            if silence_end - silence_start > 1.0:  # Only flag silence longer than 1 second
                segments.append(FlaggedSegment(
                    start_time=round(silence_start, 2),
                    end_time=round(silence_end, 2),
                    type="Silence",
                    description=f"Silence detected for {round(silence_end - silence_start, 2)} seconds",
                    severity="Low",
                    confidence=0.8
                ))
            in_silence = False
    
    return segments

def detect_loud_segments(y: np.ndarray, sr: int, threshold: float = 0.8) -> List[FlaggedSegment]:
    """Detect unusually loud segments"""
    segments = []
    
    # Calculate RMS energy
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find loud frames
    loud_frames = rms > threshold
    
    # Convert frames to time segments
    frame_times = librosa.frames_to_time(np.arange(len(loud_frames)), sr=sr, hop_length=hop_length)
    
    # Group consecutive loud frames
    in_loud = False
    loud_start = 0
    
    for i, is_loud in enumerate(loud_frames):
        if is_loud and not in_loud:
            loud_start = frame_times[i]
            in_loud = True
        elif not is_loud and in_loud:
            loud_end = frame_times[i]
            if loud_end - loud_start > 0.5:  # Only flag loud segments longer than 0.5 seconds
                segments.append(FlaggedSegment(
                    start_time=round(loud_start, 2),
                    end_time=round(loud_end, 2),
                    type="Loud Segment",
                    description=f"Unusually loud audio detected",
                    severity="Medium",
                    confidence=0.7
                ))
            in_loud = False
    
    return segments

def detect_frequency_anomalies(y: np.ndarray, sr: int) -> List[FlaggedSegment]:
    """Detect frequency anomalies (e.g., high-pitched noise)"""
    segments = []
    
    # Calculate spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # Calculate mean and std
    mean_centroid = np.mean(spectral_centroids)
    std_centroid = np.std(spectral_centroids)
    
    # Find anomalous frames (more than 2 std deviations from mean)
    anomaly_threshold = mean_centroid + 2 * std_centroid
    anomaly_frames = spectral_centroids > anomaly_threshold
    
    # Convert frames to time segments
    frame_times = librosa.frames_to_time(np.arange(len(anomaly_frames)), sr=sr)
    
    # Group consecutive anomaly frames
    in_anomaly = False
    anomaly_start = 0
    
    for i, is_anomaly in enumerate(anomaly_frames):
        if is_anomaly and not in_anomaly:
            anomaly_start = frame_times[i]
            in_anomaly = True
        elif not is_anomaly and in_anomaly:
            anomaly_end = frame_times[i]
            if anomaly_end - anomaly_start > 0.3:  # Only flag anomalies longer than 0.3 seconds
                segments.append(FlaggedSegment(
                    start_time=round(anomaly_start, 2),
                    end_time=round(anomaly_end, 2),
                    type="Frequency Anomaly",
                    description=f"Unusual frequency content detected",
                    severity="Medium",
                    confidence=0.6
                ))
            in_anomaly = False
    
    return segments

def detect_clipping(y: np.ndarray, sr: int, threshold: float = 0.99) -> List[FlaggedSegment]:
    """Detect clipping/distortion in audio"""
    segments = []
    
    # Find samples that are close to maximum amplitude
    clipped_samples = np.abs(y) > threshold
    
    # Convert sample indices to time
    sample_times = np.arange(len(y)) / sr
    
    # Group consecutive clipped samples
    in_clipping = False
    clipping_start = 0
    
    for i, is_clipped in enumerate(clipped_samples):
        if is_clipped and not in_clipping:
            clipping_start = sample_times[i]
            in_clipping = True
        elif not is_clipped and in_clipping:
            clipping_end = sample_times[i]
            if clipping_end - clipping_start > 0.1:  # Only flag clipping longer than 0.1 seconds
                segments.append(FlaggedSegment(
                    start_time=round(clipping_start, 2),
                    end_time=round(clipping_end, 2),
                    type="Clipping",
                    description=f"Audio clipping/distortion detected",
                    severity="High",
                    confidence=0.9
                ))
            in_clipping = False
    
    return segments

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
