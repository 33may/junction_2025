import React, { useState, useEffect } from 'react';
import AudioUploader from './components/AudioUploader';
import ProcessingResults from './components/ProcessingResults';
import ThemeToggle from './components/ThemeToggle';
import PersonaGallery from './components/PersonaGallery';
import DebateChat from './components/DebateChat';
import './App.css';

import johnImg from './images/john_image.png';
import aishaImg from './images/aisha_image.png';
import emilyImg from './images/emily_image.png';
import markusImg from './images/markus_image.png';

function App() {
  const [processingResults, setProcessingResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [theme, setTheme] = useState('system');
  const [debateMessages, setDebateMessages] = useState([]);
  const [isDebateActive, setIsDebateActive] = useState(false);

  const personas = [
    {
      "first_name": "John",
      "last_name": "Doe",
      "age": 45,
      "sex": "male",
      "race": "white",
      "race_option": "european",
      "race_options": "north_american",
      "country": "US",
      "city/state": "New York",
      "political_views": "very_conservative",
      "party_identification": "republican",
      "family_structure_at_16": "lived_with_parents",
      "family_income_at_16": "average",
      "fathers_highest_degree": "high_school",
      "mothers_highest_degree": "some_college",
      "marital_status": "married",
      "work_status": "employed",
      "military_service_duration": "4_10_years",
      "religion": "catholic",
      "immigrated_to_current_country": "no",
      "citizenship_status": "citizen",
      "highest_degree_received": "some_college",
      "speak_other_language": "no",
      "total_wealth": "100k_500k",
      "personality_traits": {
        "extroversion": "medium",
        "openness_to_experience": "medium",
        "conscientiousness": "high",
        "agreeableness": "medium",
        "emotional_stability": "medium",
        "aggressiveness": "medium"
      },
      "skills_and_capabilities": {
        "debate_ability": "medium",
        "critical_thinking": "high",
        "contribute_own_ideas": "medium",
        "leadership": "high",
        "resilience_under_pressure": "high",
        "teamwork": "medium",
        "creativity": "medium"
      },
      "image_url": johnImg,
      "story": "John grew up in a stable, middle-class household in New York. With a passion for community and order, he pursued a career in law enforcement. His moderate conservative views align with his commitment to preserving tradition and ensuring safety. A former military serviceman, John values discipline and responsibility. Though technology isn't his forte, he champions its use for community safety. Driven by his strong conscience and leadership skills, he's seen as a guiding figure in his neighborhood, promoting values of hard work, respect, and integrity.",
    },
    {
      "first_name": "Aisha",
      "image_url": aishaImg,
      "last_name": "Khan",
      "age": 32,
      "sex": "female",
      "race": "south_asian",
      "race_options": "pakistani",
      "country": "GB",
      "city/state": "London",
      "political_views": "very_liberal",
      "party_identification": "green",
      "family_structure_at_16": "lived_with_parents",
      "family_income_at_16": "average",
      "fathers_highest_degree": "bachelor",
      "mothers_highest_degree": "bachelor",
      "marital_status": "never_married",
      "work_status": "employed",
      "military_service_duration": "no_active_duty",
      "religion": "muslim",
      "immigrated_to_current_country": "yes",
      "citizenship_status": "citizen",
      "highest_degree_received": "master",
      "speak_other_language": "yes",
      "total_wealth": "25k_100k",
      "personality_traits": {
        "extroversion": "high",
        "openness_to_experience": "high",
        "conscientiousness": "medium",
        "agreeableness": "high",
        "emotional_stability": "medium",
        "aggressiveness": "low"
      },
      "skills_and_capabilities": {
        "debate_ability": "high",
        "critical_thinking": "medium",
        "contribute_own_ideas": "high",
        "leadership": "medium",
        "resilience_under_pressure": "medium",
        "teamwork": "high",
        "creativity": "high"
      },
      "story": "Aisha moved to London from Pakistan at a young age. Passionate about climate change and social justice, she pursued a career in environmental policy. She balances high openness with her Islamic faith, advocating for inclusive policies. Active in her community, she leverages her creativity to implement innovative projects for sustainable living. Despite occasional challenges in navigating cultural nuances, Aisha's ability to connect diverse groups fosters collaboration and understanding, helping her excel as a policy analyst."
    },
    {
      "first_name": "Carlos",
      "image_url": markusImg,
      "last_name": "Ramirez",
      "age": 60,
      "sex": "male",
      "race": "latino_hispanic",
      "race_options": "mexican",
      "country": "MX",
      "city/state": "Mexico City",
      "political_views": "moderate",
      "party_identification": "independent_pure",
      "family_structure_at_16": "guardian",
      "family_income_at_16": "low",
      "fathers_highest_degree": "no_schooling",
      "mothers_highest_degree": "primary",
      "marital_status": "divorced",
      "work_status": "retired",
      "military_service_duration": "no_active_duty",
      "religion": "catholic",
      "immigrated_to_current_country": "no",
      "citizenship_status": "citizen",
      "highest_degree_received": "high_school",
      "speak_other_language": "no",
      "total_wealth": "5k_25k",
      "personality_traits": {
        "extroversion": "low",
        "openness_to_experience": "medium",
        "conscientiousness": "high",
        "agreeableness": "low",
        "emotional_stability": "low",
        "aggressiveness": "medium"
      },
      "skills_and_capabilities": {
        "debate_ability": "low",
        "critical_thinking": "medium",
        "contribute_own_ideas": "low",
        "leadership": "medium",
        "resilience_under_pressure": "low",
        "teamwork": "medium",
        "creativity": "low"
      },
      "story": "Carlos grew up under challenging circumstances, raised by his aunt due to family instability. Despite limited educational opportunities, he became a dedicated worker, supporting his children through secondary school. Now retired, Carlos occasionally struggles with loneliness but finds solace and purpose in his faith and the local church community. Though not very vocal about politics, his pragmatic views allow him to examine issues without biases. His life experience lends wisdom, valuing hard-earned respect and honesty."
    }]

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

  const handlePersonaClick = (persona) => {
    // Handle persona click (e.g., open modal, route, or set selected state)
    // This is intentionally minimal to keep your current UX intact
    console.log('Persona clicked:', persona);
  };

  const handleStartDebate = () => {
    setIsDebateActive(true);
    setDebateMessages([]);
  };

  const handleStopDebate = () => {
    setIsDebateActive(false);
  };


  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <div className="theme-toggle-container">
            <ThemeToggle theme={theme} onThemeChange={setTheme} />
          </div>
          <div className="header-content">
            <div className="header-text">
              <h1>Audio Transcription & Analysis</h1>
              <p>Upload an audio file to get transcription and detect extremist content</p>
            </div>
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

        {processingResults && Array.isArray(personas) && personas.length > 0 && (
        <PersonaGallery personas={personas} onPersonaClick={handlePersonaClick} />
      )}

      {processingResults && (
        <div className="debate-section">
          <div className="debate-controls">
            <h3>Debate Discussion</h3>
            <div className="debate-buttons">
              {!isDebateActive ? (
                <button 
                  className="debate-start-btn"
                  onClick={handleStartDebate}
                >
                  Start Debate
                </button>
              ) : (
                <button 
                  className="debate-stop-btn"
                  onClick={handleStopDebate}
                >
                  Stop Debate
                </button>
              )}
            </div>
          </div>
          
          {isDebateActive && (
            <DebateChat 
              participants={personas.filter(p => 
                // Mock: assume first 2 personas are in debate group
                personas.indexOf(p) < 2
              )}
              messages={debateMessages}
            />
          )}
        </div>
      )}
    </div>
  );
}

export default App;
