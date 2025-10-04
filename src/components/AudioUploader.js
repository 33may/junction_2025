import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { uploadAudioFile } from '../services/api';
import './AudioUploader.css';

const AudioUploader = ({ onProcessingStart, onProcessingComplete, onError, isProcessing }) => {
  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('audio/')) {
      onError('Please upload an audio file (mp3, wav, etc.)');
      return;
    }

    // Validate file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
      onError('File size must be less than 50MB');
      return;
    }

    try {
      onProcessingStart();
      const results = await uploadAudioFile(file);
      onProcessingComplete(results, file);
    } catch (error) {
      onError(error.message || 'Failed to process audio file');
    }
  }, [onProcessingStart, onProcessingComplete, onError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.m4a', '.ogg', '.flac']
    },
    multiple: false,
    disabled: isProcessing
  });

  return (
    <div className="audio-uploader">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${isProcessing ? 'processing' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="dropzone-content">
          {isProcessing ? (
            <div className="processing-state">
              <div className="spinner"></div>
              <p>Processing audio file...</p>
            </div>
          ) : isDragActive ? (
            <div className="drag-active">
              <div className="upload-icon">📁</div>
              <p>Drop the audio file here...</p>
            </div>
          ) : (
            <div className="upload-prompt">
              <div className="upload-icon">🎵</div>
              <h3>Drop your audio file here</h3>
              <p>or click to browse</p>
              <div className="supported-formats">
                <small>Supported formats: MP3, WAV, M4A, OGG, FLAC</small>
                <small>Max file size: 50MB</small>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AudioUploader;
