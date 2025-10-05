import React, { useState, useEffect } from 'react';
import AudioUploader from './components/AudioUploader';
import ProcessingResults from './components/ProcessingResults';
import ThemeToggle from './components/ThemeToggle';
import PersonaGallery from './components/PersonaGallery';
import DebateChat from './components/DebateChat';
import DebateResults from './components/DebateResults';
import { simulateDebate } from './utils/debateSimulator';
import conversationData from './conversation.json';
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
  const [isTyping, setIsTyping] = useState(false);
  const [typingParticipant, setTypingParticipant] = useState(null);
  const [debateData, setDebateData] = useState(null);
  const [isDebateFinished, setIsDebateFinished] = useState(false);

  const personas = [
    { "image_url": emilyImg, "first_name": "Emily", "last_name": "R.", "age": 21, "sex": "female", "race": "white", "race_options": "north_american", "country": "US", "city/state": "California", "political_views": "very_liberal", "party_identification": "democrat", "residence_at_16": "west", "same_residence_since_16": "same_city", "family_structure_at_16": "lived_with_parents", "family_income_at_16": "high", "fathers_highest_degree": "doctorate", "mothers_highest_degree": "master", "mothers_work_history": "yes", "marital_status": "never_married", "work_status": "student", "military_service_duration": "no_active_duty", "religion": "none", "religion_at_16": "none", "immigrated_to_current_country": "no", "citizenship_status": "citizen", "highest_degree_received": "some_college", "speak_other_language": "yes", "total_wealth": "less_than_5k", "personality_traits": { "extroversion": "high", "openness_to_experience": "high", "conscientiousness": "medium", "agreeableness": "high", "emotional_stability": "medium", "aggressiveness": "low" }, "skills_and_capabilities": { "debate_ability": "high", "critical_thinking": "high", "contribute_own_ideas": "high", "leadership": "medium", "resilience_under_pressure": "medium", "teamwork": "high", "creativity": "high" }, "story": "Emily R. is a university student in California, active in feminist and climate justice movements. She sees the world through the lens of social justice and equality, believing that America has a responsibility to help immigrants and marginalized communities. She argues passionately and sometimes dramatically, but with a sharp mind for evidence. For Emily, free speech means protecting vulnerable groups, not giving a platform to hate." },
    { "image_url": markusImg,"first_name": "Markus", "last_name": "L.", "age": 38, "sex": "male", "race": "white", "race_options": "german", "country": "DE", "city/state": "Berlin", "political_views": "moderate", "party_identification": "other", "residence_at_16": "europe", "same_residence_since_16": "different_state", "family_structure_at_16": "lived_with_parents", "family_income_at_16": "average", "fathers_highest_degree": "bachelor", "mothers_highest_degree": "some_college", "mothers_work_history": "yes", "marital_status": "married", "work_status": "employed", "military_service_duration": "no_active_duty", "religion": "none", "religion_at_16": "catholic", "immigrated_to_current_country": "no", "citizenship_status": "citizen", "highest_degree_received": "master", "speak_other_language": "yes", "total_wealth": "100k_500k", "personality_traits": { "extroversion": "medium", "openness_to_experience": "high", "conscientiousness": "high", "agreeableness": "medium", "emotional_stability": "high", "aggressiveness": "low" }, "skills_and_capabilities": { "debate_ability": "medium", "critical_thinking": "high", "contribute_own_ideas": "medium", "leadership": "medium", "resilience_under_pressure": "high", "teamwork": "high", "creativity": "medium" }, "story": "Markus L. is a German IT professional living in Berlin. A supporter of social democracy, he values balanced policies and pragmatic compromises. He believes immigration is necessary for Europe’s future but must be managed responsibly. Markus speaks calmly, relying on facts and statistics rather than emotions. His role in debates is often that of a mediator between extremes, stressing democratic institutions and practical solutions." },
    { "image_url": aishaImg,"first_name": "Aisha", "last_name": "T.", "age": 29, "sex": "female", "race": "black_or_african", "race_options": "african_american", "country": "US", "city/state": "New York", "political_views": "liberal", "party_identification": "democrat", "residence_at_16": "northeast", "same_residence_since_16": "same_city", "family_structure_at_16": "single_parent", "family_income_at_16": "low", "fathers_highest_degree": "unknown", "mothers_highest_degree": "high_school", "mothers_work_history": "yes", "marital_status": "cohabiting", "work_status": "employed", "military_service_duration": "no_active_duty", "religion": "none", "religion_at_16": "protestant", "immigrated_to_current_country": "no", "citizenship_status": "citizen", "highest_degree_received": "bachelor", "speak_other_language": "no", "total_wealth": "5k_25k", "personality_traits": { "extroversion": "high", "openness_to_experience": "high", "conscientiousness": "medium", "agreeableness": "low", "emotional_stability": "medium", "aggressiveness": "high" }, "skills_and_capabilities": { "debate_ability": "high", "critical_thinking": "medium", "contribute_own_ideas": "high", "leadership": "high", "resilience_under_pressure": "high", "teamwork": "medium", "creativity": "high" }, "story": "Aisha T. grew up in Brooklyn in a low-income family with a single mother. She became politically active during college and joined the Black Lives Matter movement. Now working at a nonprofit focused on civil rights, she is passionate and uncompromising. Aisha often speaks in fiery, emotional terms, calling out systemic racism and sexism. She believes America needs radical change to protect marginalized communities and is ready to confront opponents head-on in debate." },
    { "image_url": johnImg,"first_name": "John", "last_name": "M.", "age": 54, "sex": "male", "race": "white", "race_options": "irish", "country": "US", "city/state": "Texas", "political_views": "very_conservative", "party_identification": "republican", "residence_at_16": "south", "same_residence_since_16": "same_state", "family_structure_at_16": "lived_with_parents", "family_income_at_16": "average", "fathers_highest_degree": "high_school", "mothers_highest_degree": "high_school", "mothers_work_history": "part_time", "marital_status": "married", "work_status": "employed", "military_service_duration": "no_active_duty", "religion": "protestant", "religion_at_16": "protestant", "immigrated_to_current_country": "no", "citizenship_status": "citizen", "highest_degree_received": "high_school", "speak_other_language": "no", "total_wealth": "25k_100k", "personality_traits": { "extroversion": "medium", "openness_to_experience": "low", "conscientiousness": "high", "agreeableness": "low", "emotional_stability": "medium", "aggressiveness": "high" }, "skills_and_capabilities": { "debate_ability": "medium", "critical_thinking": "medium", "contribute_own_ideas": "high", "leadership": "high", "resilience_under_pressure": "high", "teamwork": "medium", "creativity": "low" }, "story": "John M. is the owner of a small auto repair shop in Texas. He has worked with his hands since he was young and believes in traditional values and honest labor. He blames illegal immigrants for undercutting his business and has become a strong opponent of immigration. A vocal Trump supporter and NRA member, John speaks bluntly and often offensively, but with conviction. He sees 'hate speech' accusations as censorship and believes real patriots must defend free speech at all costs." }]

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

  const handleStartDebate = async () => {
    // Always reset to beginning
    setIsDebateActive(true);
    setDebateMessages([]);
    setIsTyping(false);
    setTypingParticipant(null);
    setIsDebateFinished(false);
    
    // Use conversation data directly
    const debateData = conversationData.debateData;
    setDebateData(debateData);
    
    // Start simulation
    await simulateDebate(
      debateData,
      (message) => {
        return new Promise((resolve) => {
          setIsTyping(true);
          setTypingParticipant(message.participantId);
          
          setTimeout(() => {
            setIsTyping(false);
            setTypingParticipant(null);
            setDebateMessages(prev => [...prev, message]);
            resolve();
          }, 8000);
        });
      },
      (systemMessage) => {
        setDebateMessages(prev => [...prev, systemMessage]);
      },
      2000
    );
    
    // Mark debate as finished when simulation completes
    setIsDebateFinished(true);
  };

  const handleStopDebate = () => {
    setIsDebateActive(false);
    setIsTyping(false);
    setTypingParticipant(null);
    setIsDebateFinished(false);
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
              <h1>Inclusive Speech Data Screening</h1>
              <p>
                A responsible tool to identify extremist views and harmful language in audio and video, enabling safer and more inclusive speech technology.
              </p>
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
              participants={personas}
              messages={debateMessages}
              isTyping={isTyping}
              typingParticipant={typingParticipant}
            />
          )}
          
          {isDebateFinished && debateData && (
            <DebateResults 
              finalResult={debateData.finalResult}
              isVisible={isDebateFinished}
            />
          )}
        </div>
      )}
    </div>
  );
}

export default App;
