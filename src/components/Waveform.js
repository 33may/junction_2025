import React, { useRef, useEffect, useState } from 'react';
import './Waveform.css';

const Waveform = ({ 
  audioFile, 
  transcription = [], 
  onTimeUpdate, 
  currentTime = 0,
  highlightedSegment = null,
  isPlaying = false,
  onPlayPause,
  duration = 0
}) => {
  const canvasRef = useRef(null);
  const audioRef = useRef(null);
  const [audioContext, setAudioContext] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [waveformData, setWaveformData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    if (audioFile) {
      loadAudioFile();
    }
  }, [audioFile]);

  useEffect(() => {
    if (waveformData && canvasRef.current) {
      // Ensure canvas is properly sized
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      const ctx = canvas.getContext('2d');
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      
      drawWaveform();
    }
  }, [waveformData, currentTime, highlightedSegment, isPlaying]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.addEventListener('timeupdate', handleAudioTimeUpdate);
      audioRef.current.addEventListener('ended', handleAudioEnded);
      return () => {
        audioRef.current?.removeEventListener('timeupdate', handleAudioTimeUpdate);
        audioRef.current?.removeEventListener('ended', handleAudioEnded);
      };
    }
  }, [audioRef.current]);

  const loadAudioFile = async () => {
    setIsLoading(true);
    try {
      const arrayBuffer = await audioFile.arrayBuffer();
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const buffer = await audioCtx.decodeAudioData(arrayBuffer);
      
      setAudioContext(audioCtx);
      setAudioBuffer(buffer);
      
      // Generate waveform data
      const data = generateWaveformData(buffer);
      setWaveformData(data);
    } catch (error) {
      console.error('Error loading audio file:', error);
      // Generate fallback waveform data
      const fallbackData = generateFallbackWaveform();
      setWaveformData(fallbackData);
    } finally {
      setIsLoading(false);
    }
  };

  const generateFallbackWaveform = () => {
    // Generate a simple sine wave pattern as fallback
    const data = [];
    for (let i = 0; i < 1000; i++) {
      data.push(Math.abs(Math.sin(i * 0.1)) * 0.5 + Math.random() * 0.3);
    }
    return data;
  };

  const generateWaveformData = (buffer) => {
    const channelData = buffer.getChannelData(0);
    const samplesPerPixel = Math.floor(channelData.length / 1000); // 1000px width
    const data = [];

    for (let i = 0; i < 1000; i++) {
      const start = i * samplesPerPixel;
      const end = Math.min(start + samplesPerPixel, channelData.length);
      
      let sum = 0;
      for (let j = start; j < end; j++) {
        sum += Math.abs(channelData[j]);
      }
      
      data.push(sum / (end - start));
    }

    return data;
  };

  const drawWaveform = () => {
    const canvas = canvasRef.current;
    if (!canvas || !waveformData) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas with background color
    ctx.fillStyle = 'var(--waveform-bg, #f8f9fa)';
    ctx.fillRect(0, 0, width, height);
    
    // Set up drawing parameters
    const centerY = height / 2;
    const maxAmplitude = Math.max(...waveformData);
    
    // Draw waveform background
    ctx.fillStyle = 'var(--waveform-bg, #f8f9fa)';
    ctx.fillRect(0, 0, width, height);
    
    // Draw extremist segments first (background) - only if we have audio buffer
    if (audioBuffer) {
      transcription.forEach((sentence) => {
        if (sentence.is_extremist) {
          const startX = (sentence.start_time / audioBuffer.duration) * width;
          const endX = (sentence.end_time / audioBuffer.duration) * width;
          
          ctx.fillStyle = 'var(--extremist-bg, rgba(220, 53, 69, 0.1))';
          ctx.fillRect(startX, 0, endX - startX, height);
        }
      });
    }
    
    // Draw waveform
    ctx.beginPath();
    ctx.strokeStyle = 'var(--waveform-color, #4a90e2)';
    ctx.lineWidth = 2;
    
    waveformData.forEach((amplitude, index) => {
      const x = (index / waveformData.length) * width;
      const y = centerY - (amplitude / maxAmplitude) * centerY;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw mirrored waveform below center
    ctx.beginPath();
    waveformData.forEach((amplitude, index) => {
      const x = (index / waveformData.length) * width;
      const y = centerY + (amplitude / maxAmplitude) * centerY;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw current time indicator - only if we have audio buffer
    if (audioBuffer && audioBuffer.duration > 0) {
      const progress = currentTime / audioBuffer.duration;
      const currentX = progress * width;
      
      ctx.beginPath();
      ctx.strokeStyle = 'var(--progress-color, #ff6b6b)';
      ctx.lineWidth = 3;
      ctx.moveTo(currentX, 0);
      ctx.lineTo(currentX, height);
      ctx.stroke();
      
      // Draw progress fill
      ctx.fillStyle = 'var(--progress-fill, rgba(255, 107, 107, 0.2))';
      ctx.fillRect(0, 0, currentX, height);
    }
    
    // Draw highlighted segments - only if we have audio buffer
    if (highlightedSegment && audioBuffer) {
      const startX = (highlightedSegment.start_time / audioBuffer.duration) * width;
      const endX = (highlightedSegment.end_time / audioBuffer.duration) * width;
      
      ctx.fillStyle = highlightedSegment.is_extremist ? 'var(--highlight-extremist, rgba(255, 0, 0, 0.3))' : 'var(--highlight-normal, rgba(255, 255, 0, 0.3))';
      ctx.fillRect(startX, 0, endX - startX, height);
    }
  };

  const handleCanvasClick = (event) => {
    if (!audioBuffer) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const clickTime = (x / canvas.width) * audioBuffer.duration;
    
    if (onTimeUpdate) {
      onTimeUpdate(clickTime);
    }
  };

  const handleMouseDown = (event) => {
    setIsDragging(true);
    handleCanvasClick(event);
  };

  const handleMouseMove = (event) => {
    if (isDragging) {
      handleCanvasClick(event);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleAudioTimeUpdate = () => {
    if (audioRef.current && onTimeUpdate) {
      onTimeUpdate(audioRef.current.currentTime);
    }
  };

  const handleAudioEnded = () => {
    if (onPlayPause) {
      onPlayPause(false);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (isLoading) {
    return (
      <div className="waveform-container">
        <div className="waveform-loading">
          <div className="spinner"></div>
          <p>Loading waveform...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="waveform-container">
      <div className="waveform-header">
        <h3>Audio Player</h3>
        <div className="waveform-controls">
          <button 
            className="play-pause-btn"
            onClick={() => onPlayPause && onPlayPause(!isPlaying)}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <span className="time-display">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>
      </div>
      
      <div className="waveform-wrapper">
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="waveform-canvas"
        />
      </div>
      
      <audio
        ref={audioRef}
        src={audioFile ? URL.createObjectURL(audioFile) : undefined}
        preload="metadata"
      />
    </div>
  );
};

export default Waveform;
