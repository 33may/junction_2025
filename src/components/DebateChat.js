import React, { useRef, useEffect, useState } from 'react';
import './DebateChat.css';

const DebateChat = ({ participants = [], messages = [], isTyping = false, typingParticipant = null }) => {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const getParticipantInfo = (participantId) => {
    // Try to match by first name (most common case)
    const byFirstName = participants.find(p => 
      p.first_name.toLowerCase() === participantId.toLowerCase()
    );
    if (byFirstName) return byFirstName;
    
    // Try to match by full name
    const byFullName = participants.find(p => 
      `${p.first_name} ${p.last_name}`.toLowerCase() === participantId.toLowerCase()
    );
    if (byFullName) return byFullName;
    
    // Try to match by ID if it exists
    const byId = participants.find(p => 
      p.id && p.id.toLowerCase() === participantId.toLowerCase()
    );
    if (byId) return byId;
    
    // Try partial matching
    const byPartial = participants.find(p => 
      participantId.toLowerCase().includes(p.first_name.toLowerCase()) ||
      p.first_name.toLowerCase().includes(participantId.toLowerCase())
    );
    if (byPartial) return byPartial;
    
    return null;
  };

  const TypingIndicator = ({ participant }) => {
    return (
      <div className="message typing-message">
        <div className="message-avatar">
          <img 
            src={participant?.image_url || participant?.image || participant?.avatar || 'https://via.placeholder.com/32x32?text=?'} 
            alt={`${participant?.first_name || 'Unknown'} ${participant?.last_name || ''}`}
            className="message-avatar-image"
          />
        </div>
        <div className="message-content">
          <div className="message-header">
            <span className="message-sender">
              {participant ? `${participant.first_name} ${participant.last_name}` : 'Unknown'}
            </span>
          </div>
          <div className="message-text typing-indicator">
            <span className="typing-dot"></span>
            <span className="typing-dot"></span>
            <span className="typing-dot"></span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="debate-chat">
      <div className="debate-chat-header">
        <h3>Debate Discussion</h3>
        <div className="debate-participants">
          {participants.map((participant, index) => (
            <div key={participant.id || `${participant.first_name}-${participant.last_name}`} 
                 className="participant-avatar">
              <img 
                src={participant.image_url || participant.image || participant.avatar || 'https://via.placeholder.com/40x40?text=?'} 
                alt={`${participant.first_name} ${participant.last_name}`}
                className="avatar-image"
              />
              <span className="avatar-name">{participant.first_name}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="debate-messages">
        {messages.length === 0 ? (
          <div className="no-messages">
            <p>The debate is starting... Participants will begin discussing shortly.</p>
          </div>
        ) : (
          <>
            {messages.map((message, index) => {
              if (message.type === 'system') {
                return (
                  <div key={message.id || index} className="system-message">
                    <div className="system-message-content">
                      <div className="system-message-text">
                        {message.text.split('\n').map((line, i) => (
                          <div key={i}>{line}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              }

              const participant = getParticipantInfo(message.participantId);
              return (
                <div key={message.id || index} className={`message ${message.type || 'agent'}`}>
                  <div className="message-avatar">
                    <img 
                      src={participant?.image_url || participant?.image || participant?.avatar || 'https://via.placeholder.com/32x32?text=?'} 
                      alt={`${participant?.first_name || 'Unknown'} ${participant?.last_name || ''}`}
                      className="message-avatar-image"
                    />
                  </div>
                  <div className="message-content">
                    <div className="message-header">
                      <span className="message-sender">
                        {participant ? `${participant.first_name} ${participant.last_name}` : 'Unknown'}
                      </span>
                    </div>
                    <div className="message-text">
                      {message.text}
                    </div>
                    {(message.hate_speech !== undefined || message.extremism !== undefined) && (
                      <div className="message-votes">
                        <span className={`vote-badge ${message.hate_speech ? 'hate-speech-true' : 'hate-speech-false'}`}>
                          Hate Speech: {message.hate_speech ? 'True' : 'False'}
                        </span>
                        <span className={`vote-badge ${message.extremism ? 'extremism-true' : 'extremism-false'}`}>
                          Extremism: {message.extremism ? 'True' : 'False'}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
            {isTyping && typingParticipant && (
              <TypingIndicator participant={getParticipantInfo(typingParticipant)} />
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};

export default DebateChat;
