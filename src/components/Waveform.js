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
  duration = 0,
  onSentenceHover
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

  // Redraw waveform when theme changes
  useEffect(() => {
    const handleThemeChange = () => {
      if (waveformData && canvasRef.current) {
        drawWaveform();
      }
    };

    // Listen for theme changes
    const observer = new MutationObserver(handleThemeChange);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme']
    });

    return () => observer.disconnect();
  }, [waveformData]);

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
    const samplesPerPixel = Math.floor(channelData.length / 1500) || 1; // higher resolution
    const raw = [];
    const frames = Math.floor(channelData.length / samplesPerPixel);
    for (let i = 0; i < frames; i++) {
      const start = i * samplesPerPixel;
      const end = Math.min(start + samplesPerPixel, channelData.length);
      let sum = 0;
      for (let j = start; j < end; j++) {
        sum += Math.abs(channelData[j]);
      }
      raw.push(sum / (end - start));
    }

    // Normalize against a high percentile to avoid single-sample spikes dominating
    const sorted = [...raw].sort((a, b) => a - b);
    const pIndex = Math.floor(sorted.length * 0.98);
    const reference = Math.max(sorted[pIndex] || 1e-4, 1e-4);

    // Map to 0..1 and clamp so quiet audio still looks visible
    return raw.map(v => {
      const norm = v / reference;
      // use a mild gamma to boost lower amplitudes visually
      return Math.min(1, Math.pow(norm, 0.7));
    });
  };
  
  const drawWaveform = () => {
    const canvas = canvasRef.current;
    if (!canvas || !waveformData) return;
    const ctx = canvas.getContext('2d');

    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.width / dpr;
    const cssHeight = canvas.height / dpr;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(dpr, dpr);

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

    // stronger, more visible base colors
    const bgColor = 'transparent'; // card background already covers it
    const unplayedColor = isDark ? 'rgba(255,255,255,0.18)' : 'rgba(15,23,42,0.18)';
    const playedStart = isDark ? '#7dd3fc' : '#60a5fa';
    const playedEnd = isDark ? '#60a5fa' : '#1e40af';
    const highlightExtremist = isDark ? 'rgba(248,81,73,0.18)' : 'rgba(239,68,68,0.16)';

    // Fill background to avoid artifacts on some browsers
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    const centerY = cssHeight / 2;
    // Scale bars to 70% of total canvas height (so peak-to-trough = 70% of canvas)
    const desiredFullHeightRatio = 0.7; // 70%
    const maxBarHeight = (cssHeight * desiredFullHeightRatio) / 2;
    // keep a tiny padding so bars never touch edges
    const paddingY = Math.max(2, centerY - maxBarHeight);

    // Draw extremist overlays first (subtle)
    if (audioBuffer && transcription && transcription.length) {
      transcription.forEach((sentence) => {
        if (sentence.is_extremist) {
          const startX = (sentence.start_time / audioBuffer.duration) * cssWidth;
          const endX = (sentence.end_time / audioBuffer.duration) * cssWidth;
          ctx.fillStyle = highlightExtremist;
          ctx.fillRect(startX, 0, Math.max(1, endX - startX), cssHeight);
        }
      });
    }

    // Bars configuration
    const approxBarWidth = 3;
    const gapRatio = 0.12; // reduce gap to make bars bolder
    const maxBars = Math.floor(cssWidth / (approxBarWidth + 1));
    const bars = Math.min(maxBars, waveformData.length);
    const step = waveformData.length / bars;
    const barGap = Math.max(1, Math.floor((cssWidth / bars) * gapRatio));
    const barWidth = Math.max(2, Math.floor((cssWidth / bars) - barGap));

    // Precompute progress X
    const progressX = (audioBuffer && audioBuffer.duration > 0) ? (currentTime / audioBuffer.duration) * cssWidth : 0;

    // Draw unplayed bars (entire waveform) with higher visibility so right side is legible
    ctx.lineWidth = Math.max(2, barWidth);
    ctx.lineCap = 'round';
    ctx.strokeStyle = unplayedColor;
    ctx.shadowBlur = 0;
    for (let i = 0; i < bars; i++) {
      const dataIdx = Math.floor(i * step);
      const amp = waveformData[dataIdx] || 0;
      // mild nonlinear mapping to make small amplitudes more visible
      const mapped = Math.pow(amp, 0.7);
      const h = Math.max(2, mapped * maxBarHeight);
      const x = i * (barWidth + barGap) + barGap / 2;
      ctx.beginPath();
      ctx.moveTo(x + barWidth / 2, centerY - h);
      ctx.lineTo(x + barWidth / 2, centerY + h);
      ctx.stroke();
    }

    // Draw played bars (left of progress) with stronger gradient and glow
    const grad = ctx.createLinearGradient(0, 0, cssWidth, 0);
    grad.addColorStop(0, playedStart);
    grad.addColorStop(1, playedEnd);
    ctx.strokeStyle = grad;
    ctx.shadowColor = isDark ? 'rgba(96,165,250,0.22)' : 'rgba(37,99,235,0.2)';
    ctx.shadowBlur = 12;
    ctx.lineWidth = Math.max(2, barWidth);
    ctx.lineCap = 'round';
    for (let i = 0; i < bars; i++) {
      const x = i * (barWidth + barGap) + barGap / 2;
      const cx = x + barWidth / 2;
      if (cx > progressX) continue;
      const dataIdx = Math.floor(i * step);
      const amp = waveformData[dataIdx] || 0;
      const mapped = Math.pow(amp, 0.7);
      const h = Math.max(2, mapped * maxBarHeight);
      ctx.beginPath();
      ctx.moveTo(cx, centerY - h);
      ctx.lineTo(cx, centerY + h);
      ctx.stroke();
    }
    ctx.shadowBlur = 0;

    // subtle center line
    ctx.beginPath();
    ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.04)';
    ctx.lineWidth = 1;
    ctx.moveTo(0, centerY);
    ctx.lineTo(cssWidth, centerY);
    ctx.stroke();

    // Playhead
    if (audioBuffer && audioBuffer.duration > 0) {
      const currentX = progressX;
      ctx.beginPath();
      ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.95)' : 'rgba(0,0,0,0.9)';
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.moveTo(currentX, 6);
      ctx.lineTo(currentX, cssHeight - 6);
      ctx.stroke();

      // playhead knob
      ctx.beginPath();
      ctx.fillStyle = isDark ? '#ffffff' : '#000000';
      ctx.globalAlpha = 0.95;
      ctx.arc(currentX, 8, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Highlighted segment overlay (stronger)
    if (highlightedSegment && audioBuffer) {
      const startX = (highlightedSegment.start_time / audioBuffer.duration) * cssWidth;
      const endX = (highlightedSegment.end_time / audioBuffer.duration) * cssWidth;
      ctx.fillStyle = highlightedSegment.is_extremist
        ? (isDark ? 'rgba(248,81,73,0.22)' : 'rgba(239,68,68,0.18)')
        : (isDark ? 'rgba(99,102,241,0.18)' : 'rgba(99,102,241,0.14)');
      ctx.fillRect(startX, 0, Math.max(1, endX - startX), cssHeight);
    }

    ctx.restore();
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
    if (isDragging) {
      handleCanvasClick(event);
    } else {
      // Handle hover for sentence highlighting
      handleCanvasHover(event);
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleMouseLeave = () => {
    if (onSentenceHover) {
      onSentenceHover(null);
    }
  };

  const handleCanvasHover = (event) => {
    if (!audioBuffer || !transcription.length) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * window.devicePixelRatio;
    const hoverTime = (x / canvas.width) * audioBuffer.duration;
    
    // Find the sentence that contains this time
    const hoveredSentence = transcription.find(sentence => 
      hoverTime >= sentence.start_time && hoverTime <= sentence.end_time
    );
    
    // Update highlighted segment if it changed
    if (hoveredSentence !== highlightedSegment && onSentenceHover) {
      onSentenceHover(hoveredSentence);
    }
  };

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
          onMouseLeave={handleMouseLeave}
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