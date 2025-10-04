import React, { useState, useRef } from 'react';
import Waveform from './Waveform';
import './ProcessingResults.css';

const ProcessingResults = ({ results, audioFile }) => {
  const [currentTime, setCurrentTime] = useState(0);
  const [highlightedSegment, setHighlightedSegment] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleTimeUpdate = (time) => {
    setCurrentTime(time);
  };

  const handlePlayPause = (playing) => {
    setIsPlaying(playing);
  };

  const handleSentenceHover = (sentence) => {
    setHighlightedSegment(sentence);
  };

  const handleSentenceLeave = () => {
    setHighlightedSegment(null);
  };

  const handleSentenceClick = (sentence) => {
    handleTimeUpdate(sentence.start_time);
  };

  return (
    <div className="processing-results">
      <h2>Audio Analysis Results</h2>
      
      <div className="summary">
        <div className="summary-item">
          <span className="label">Total Duration:</span>
          <span className="value">{formatTime(results.total_duration)}</span>
        </div>
        <div className="summary-item">
          <span className="label">Sentences:</span>
          <span className="value">{results.transcription?.length || 0}</span>
        </div>
        <div className="summary-item">
          <span className="label">Extremist Content:</span>
          <span className="value extremist-count">
            {results.transcription?.filter(s => s.is_extremist).length || 0}
          </span>
        </div>
        <div className="summary-item">
          <span className="label">Processing Time:</span>
          <span className="value">{results.processing_time}s</span>
        </div>
      </div>

      {/* Waveform Player */}
      {audioFile && results.transcription && (
        <div className="waveform-section">
          <Waveform
            audioFile={audioFile}
            transcription={results.transcription}
            onTimeUpdate={handleTimeUpdate}
            currentTime={currentTime}
            highlightedSegment={highlightedSegment}
            isPlaying={isPlaying}
            onPlayPause={handlePlayPause}
            duration={results.total_duration}
          />
        </div>
      )}

      {/* Transcription */}
      {results.transcription && results.transcription.length > 0 && (
        <div className="transcription-section">
          <h3>Transcription</h3>
          <div className="transcription-content">
            {results.transcription.map((sentence, index) => (
              <div
                key={index}
                className={`transcription-sentence ${sentence.is_extremist ? 'extremist' : ''}`}
                onMouseEnter={() => handleSentenceHover(sentence)}
                onMouseLeave={handleSentenceLeave}
                onClick={() => handleSentenceClick(sentence)}
              >
                <div className="sentence-time">
                  {formatTime(sentence.start_time)} - {formatTime(sentence.end_time)}
                </div>
                <div className="sentence-text">
                  {sentence.text}
                </div>
                {sentence.is_extremist && (
                  <div className="extremist-indicator">
                    Extremist Content Detected
                    {sentence.extremist_reason && (
                      <div className="extremist-reason">
                        {sentence.extremist_reason}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProcessingResults;
