import React, { useRef, useEffect, useState } from 'react';
import './Waveform.css';

const Waveform = ({
  audioFile,
  transcription = [],
  onTimeUpdate,
  currentTime = 0,
  highlightedSegment = null,
  isPlaying: parentIsPlaying,
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
  const [localDuration, setLocalDuration] = useState(duration);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioSrc, setAudioSrc] = useState();

  // Load audio file and decode for waveform
  useEffect(() => {
    if (audioFile) {
      loadAudioFile();
    }
  }, [audioFile]);

  // Draw waveform when data or state changes
  useEffect(() => {
    if (waveformData && canvasRef.current) {
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      const ctx = canvas.getContext('2d');
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      drawWaveform();
    }
  }, [waveformData, currentTime, highlightedSegment, isPlaying]);

  // Attach audio event listeners
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.addEventListener('timeupdate', handleAudioTimeUpdate);
    audio.addEventListener('ended', handleAudioEnded);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    return () => {
      audio.removeEventListener('timeupdate', handleAudioTimeUpdate);
      audio.removeEventListener('ended', handleAudioEnded);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
    };
  }, [audioRef.current]);

  // Sync local and parent isPlaying
  useEffect(() => {
    if (typeof parentIsPlaying === 'boolean') {
      setIsPlaying(parentIsPlaying);
    }
  }, [parentIsPlaying]);

  // Play/Pause audio when isPlaying changes
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.play();
    } else {
      audio.pause();
    }
  }, [isPlaying, audioSrc]);

  // Seek audio when currentTime changes (from canvas click)
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (Math.abs(audio.currentTime - currentTime) > 0.1) {
      audio.currentTime = currentTime;
    }
  }, [currentTime]);

  useEffect(() => {
    if (audioFile) {
      if (typeof audioFile === 'string') {
        setAudioSrc(audioFile);
      } else {
        const url = URL.createObjectURL(audioFile);
        setAudioSrc(url);
        return () => URL.revokeObjectURL(url);
      }
    } else {
      setAudioSrc(undefined);
    }
  }, [audioFile]);

  const loadAudioFile = async () => {
    setIsLoading(true);
    try {
      let arrayBuffer;
      if (typeof audioFile === 'string') {
        const response = await fetch(audioFile);
        arrayBuffer = await response.arrayBuffer();
      } else {
        arrayBuffer = await audioFile.arrayBuffer();
      }
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const buffer = await audioCtx.decodeAudioData(arrayBuffer);
      setAudioContext(audioCtx);
      setAudioBuffer(buffer);
      setLocalDuration(buffer.duration);
      setWaveformData(generateWaveformData(buffer));
    } catch (error) {
      console.error('Error loading audio file:', error);
      setWaveformData(generateFallbackWaveform());
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadedMetadata = () => {
    const audio = audioRef.current;
    if (audio && audio.duration && !localDuration) {
      setLocalDuration(audio.duration);
    }
  };

  const generateFallbackWaveform = () => {
    const data = [];
    for (let i = 0; i < 1000; i++) {
      data.push(Math.abs(Math.sin(i * 0.1)) * 0.5 + Math.random() * 0.3);
    }
    return data;
  };

  const generateWaveformData = (buffer) => {
    const channelData = buffer.getChannelData(0);
    const samplesPerPixel = Math.floor(channelData.length / 1000);
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
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#222831'; // dark background for contrast
    ctx.fillRect(0, 0, width, height);
    const centerY = height / 2;
    const maxAmplitude = Math.max(...waveformData);

    // Draw extremist segments
    if (audioBuffer) {
      transcription.forEach((sentence) => {
        if (sentence.is_extremist) {
          const startX = (sentence.start_time / audioBuffer.duration) * width;
          const endX = (sentence.end_time / audioBuffer.duration) * width;
          ctx.fillStyle = 'rgba(220, 53, 69, 0.2)';
          ctx.fillRect(startX, 0, endX - startX, height);
        }
      });
    }

    // Draw waveform
    ctx.beginPath();
    ctx.strokeStyle = '#00adb5'; // cyan for waveform
    ctx.lineWidth = 2;
    waveformData.forEach((amplitude, index) => {
      const x = (index / waveformData.length) * width;
      const y = centerY - (amplitude / maxAmplitude) * centerY;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw mirrored waveform
    ctx.beginPath();
    waveformData.forEach((amplitude, index) => {
      const x = (index / waveformData.length) * width;
      const y = centerY + (amplitude / maxAmplitude) * centerY;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw current time indicator
    if (audioBuffer && audioBuffer.duration > 0) {
      const progress = currentTime / audioBuffer.duration;
      const currentX = progress * width;
      ctx.beginPath();
      ctx.strokeStyle = '#ff6b6b';
      ctx.lineWidth = 3;
      ctx.moveTo(currentX, 0);
      ctx.lineTo(currentX, height);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255, 107, 107, 0.2)';
      ctx.fillRect(0, 0, currentX, height);
    }

    // Draw highlighted segments
    if (highlightedSegment && audioBuffer) {
      const startX = (highlightedSegment.start_time / audioBuffer.duration) * width;
      const endX = (highlightedSegment.end_time / audioBuffer.duration) * width;
      ctx.fillStyle = highlightedSegment.is_extremist
        ? 'rgba(255, 0, 0, 0.3)'
        : 'rgba(255, 255, 0, 0.3)';
      ctx.fillRect(startX, 0, endX - startX, height);
    }
  };

  // Seek audio on canvas click/drag
  const handleCanvasClick = (event) => {
    if (!audioBuffer) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * window.devicePixelRatio;
    const clickTime = (x / canvas.width) * audioBuffer.duration;
    if (onTimeUpdate) onTimeUpdate(clickTime);
  };

  const handleMouseDown = (event) => {
    setIsDragging(true);
    handleCanvasClick(event);
  };

  const handleMouseMove = (event) => {
    if (isDragging) handleCanvasClick(event);
  };

  const handleMouseUp = () => setIsDragging(false);

  // Sync currentTime from audio element
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const updateTime = () => {
      if (onTimeUpdate) onTimeUpdate(audio.currentTime);
    };
    audio.addEventListener('timeupdate', updateTime);
    return () => {
      audio.removeEventListener('timeupdate', updateTime);
    };
  }, [onTimeUpdate]);

  const handleAudioEnded = () => {
    if (onPlayPause) onPlayPause(false);
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

  const handlePlayPause = () => {
    if (onPlayPause) {
      onPlayPause(!isPlaying);
    } else {
      setIsPlaying((prev) => !prev);
    }
  };

  // Add missing handleAudioTimeUpdate function
  const handleAudioTimeUpdate = () => {
    const audio = audioRef.current;
    if (audio && onTimeUpdate) {
      onTimeUpdate(audio.currentTime);
    }
  };

  return (
    <div className="waveform-container">
      <div className="waveform-header">
        <h3>Audio Player</h3>
        <div className="waveform-controls">
          <button
            className="play-pause-btn"
            onClick={handlePlayPause}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <span className="time-display">
            {formatTime(currentTime)} / {formatTime(localDuration)}
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
        src={audioSrc}
        preload="metadata"
        style={{ display: 'none' }}
      />
    </div>
  );
};

export default Waveform;