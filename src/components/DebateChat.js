import React, { useRef, useEffect } from 'react';
import './DebateChat.css';

const DebateChat = ({ participants = [], messages = [] }) => {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getParticipantInfo = (participantId) => {
    return participants.find(p => 
      (p.id || `${p.first_name}-${p.last_name}`) === participantId
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
          messages.map((message, index) => {
            const participant = getParticipantInfo(message.participantId);
            return (
              <div key={index} className={`message ${message.type || 'user'}`}>
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
                    <span className="message-time">
                      {message.timestamp ? new Date(message.timestamp).toLocaleTimeString() : 'Now'}
                    </span>
                  </div>
                  <div className="message-text">
                    {message.text}
                  </div>
                  {message.reasoning && (
                    <div className="message-reasoning">
                      <strong>Reasoning:</strong> {message.reasoning}
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};

export default DebateChat;
