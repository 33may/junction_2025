import React from 'react';
import './DebateResults.css';

const DebateResults = ({ finalResult, isVisible }) => {
  if (!isVisible || !finalResult) {
    return null;
  }

  const { final_hate_speech, final_extremism, final_votes } = finalResult;

  return (
    <div className="debate-results">
      <div className="debate-results-header">
        <h3>Debate Results</h3>
      </div>
      
      <div className="debate-results-content">
        <div className="final-verdict">
          <h4>Final Verdict</h4>
          <div className="verdict-badges">
            <div className={`verdict-badge ${final_hate_speech ? 'hate-speech-true' : 'hate-speech-false'}`}>
              Hate Speech: {final_hate_speech ? 'True' : 'False'}
            </div>
            <div className={`verdict-badge ${final_extremism ? 'extremism-true' : 'extremism-false'}`}>
              Extremism: {final_extremism ? 'True' : 'False'}
            </div>
          </div>
        </div>

        {final_votes && (
          <div className="vote-summary">
            <h4>Vote Summary</h4>
            <div className="vote-stats">
              <div className="vote-stat">
                <span className="vote-label">Hate Speech:</span>
                <span className="vote-count">
                  {final_votes.hate_speech.true} True, {final_votes.hate_speech.false} False
                </span>
              </div>
              <div className="vote-stat">
                <span className="vote-label">Extremism:</span>
                <span className="vote-count">
                  {final_votes.extremism.true} True, {final_votes.extremism.false} False
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DebateResults;
