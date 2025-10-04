import React, { useState, useEffect } from 'react';
import AudioUploader from './components/AudioUploader';
import ProcessingResults from './components/ProcessingResults';
import ThemeToggle from './components/ThemeToggle';
import './App.css';

function App() {
  const [processingResults, setProcessingResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [theme, setTheme] = useState('system');

  useEffect(() => {
    // Apply theme on mount
    applyTheme(theme);
  }, [theme]);

  const applyTheme = (selectedTheme) => {
    const root = document.documentElement;
    
    if (selectedTheme === 'system') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
    } else {
      root.setAttribute('data-theme', selectedTheme);
    }
  };

  const handleProcessingComplete = (results, file) => {
    setProcessingResults(results);
    setAudioFile(file);
    setIsProcessing(false);
    setError(null);
  };

  const handleProcessingStart = () => {
    setIsProcessing(true);
    setError(null);
    setProcessingResults(null);
    setAudioFile(null);
  };

  const handleError = (errorMessage) => {
    setError(errorMessage);
    setIsProcessing(false);
    setProcessingResults(null);
    setAudioFile(null);
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <div className="header-content">
            <div className="header-text">
              <h1>Audio Transcription & Analysis</h1>
              <p>Upload an audio file to get transcription and detect extremist content</p>
            </div>
            <ThemeToggle theme={theme} onThemeChange={setTheme} />
          </div>
        </header>

        {!processingResults && !isProcessing && (
          <div className="upload-section">
            <AudioUploader
              onProcessingStart={handleProcessingStart}
              onProcessingComplete={handleProcessingComplete}
              onError={handleError}
              isProcessing={isProcessing}
            />
          </div>
        )}

        {error && (
          <div className="error">
            <strong>Error:</strong> {error}
          </div>
        )}

        {isProcessing && (
          <div className="loading">
            <h3>Processing audio file...</h3>
            <p>This may take a few moments depending on the file size.</p>
          </div>
        )}

        {processingResults && (
          <div className="results-section">
            <ProcessingResults results={processingResults} audioFile={audioFile} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
