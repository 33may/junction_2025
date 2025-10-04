import React, { useState } from 'react';
import './PersonaGallery.css';

/** Single persona card with 16:9 image on the left and details on the right */
const PersonaCard = ({ persona, onClick, isInDiscussionGroup, onToggleDiscussionGroup }) => {
  // Use provided image URL or fallback placeholder
  const imgSrc = persona.image_url || persona.image || persona.avatar || 'https://via.placeholder.com/640x360?text=Persona';

  return (
    <div className="persona-card" onClick={() => onClick?.(persona)} role="button" tabIndex={0}
         onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onClick?.(persona)}>
      <div className="persona-card-left">
        <div className="persona-image-wrapper">
          <img src={imgSrc} alt={`${persona.first_name} ${persona.last_name}`} className="persona-image" />
        </div>
      </div>

      <div className="persona-card-right">
        <div className="persona-header">
          <h3 className="persona-name">
            {persona.first_name} {persona.last_name}
          </h3>
          <div className="persona-meta">
            <span className="meta-chip">{persona.age} y/o</span>
            {persona.sex && <span className="meta-chip">{persona.sex}</span>}
            {persona.country && <span className="meta-chip">{persona.country}</span>}
            {persona['city/state'] && <span className="meta-chip">{persona['city/state']}</span>}
          </div>
        </div>

        <div className="persona-info-grid">
          <div className="persona-info-item">
            <div className="persona-section-title">Race</div>
            <div className="persona-chips">
              <span className="meta-chip">{persona.race}</span>
              {persona.race_option && <span className="meta-chip">{persona.race_option}</span>}
              {persona.race_options && typeof persona.race_options === 'string' && <span className="meta-chip">{persona.race_options}</span>}
            </div>
          </div>

          <div className="persona-info-item">
            <div className="persona-section-title">Politics</div>
            <div className="persona-chips">
              <span className="meta-chip">{persona.political_views}</span>
              {persona.party_identification && <span className="meta-chip">{persona.party_identification}</span>}
            </div>
          </div>

          <div className="persona-info-item">
            <div className="persona-section-title">Education</div>
            <div className="persona-chips">
              <span className="meta-chip">{persona.highest_degree_received || '—'}</span>
            </div>
          </div>

          <div className="persona-info-item">
            <div className="persona-section-title">Work</div>
            <div className="persona-chips">
              <span className="meta-chip">{persona.work_status || '—'}</span>
              {persona.military_service_duration && <span className="meta-chip">military: {persona.military_service_duration}</span>}
            </div>
          </div>
        </div>

        {persona.personality_traits && (
          <div className="persona-section">
            <div className="persona-section-title">Personality</div>
            <div className="persona-chips">
              {Object.entries(persona.personality_traits).map(([k, v]) => (
                <span className="meta-chip" key={k}>{k.replaceAll('_',' ')}: {v}</span>
              ))}
            </div>
          </div>
        )}

        {persona.skills_and_capabilities && (
          <div className="persona-section">
            <div className="persona-section-title">Skills</div>
            <div className="persona-chips">
              {Object.entries(persona.skills_and_capabilities).map(([k, v]) => (
                <span className="meta-chip" key={k}>{k.replaceAll('_',' ')}: {v}</span>
              ))}
            </div>
          </div>
        )}

        {persona.story && (
          <div className="persona-story">
            {persona.story}
          </div>
        )}

        <div className="persona-actions">
          <button 
            className={`discussion-group-btn ${isInDiscussionGroup ? 'in-group' : 'not-in-group'}`}
            onClick={(e) => {
              e.stopPropagation();
              onToggleDiscussionGroup?.(persona);
            }}
          >
            {isInDiscussionGroup ? 'Remove from Debate Group' : 'Add to Discussion Group'}
          </button>
        </div>
      </div>
    </div>
  );
};

/** Gallery wrapper */
const PersonaGallery = ({ personas = [], onPersonaClick }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [discussionGroupMembers, setDiscussionGroupMembers] = useState(new Set());

  const handlePrevious = () => {
    setCurrentIndex((prev) => (prev === 0 ? personas.length - 1 : prev - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prev) => (prev === personas.length - 1 ? 0 : prev + 1));
  };

  const handleToggleDiscussionGroup = (persona) => {
    const personaId = persona.id || `${persona.first_name}-${persona.last_name}`;
    setDiscussionGroupMembers(prev => {
      const newSet = new Set(prev);
      if (newSet.has(personaId)) {
        newSet.delete(personaId);
      } else {
        newSet.add(personaId);
      }
      return newSet;
    });
  };

  const currentPersona = personas[currentIndex];
  const isInDiscussionGroup = currentPersona ? 
    discussionGroupMembers.has(currentPersona.id || `${currentPersona.first_name}-${currentPersona.last_name}`) : 
    false;

  return (
    <div className="persona-gallery">
      <div className="persona-gallery-header">
        <h3>AI Characters</h3>
      </div>

      <div className="persona-grid">
        {currentPersona && (
          <PersonaCard 
            key={currentPersona.id || `${currentPersona.first_name}-${currentPersona.last_name}-${currentIndex}`} 
            persona={currentPersona} 
            onClick={onPersonaClick}
            isInDiscussionGroup={isInDiscussionGroup}
            onToggleDiscussionGroup={handleToggleDiscussionGroup}
          />
        )}
      </div>

      <div className="persona-gallery-controls">
        <button 
          className="gallery-nav-btn prev-btn" 
          onClick={handlePrevious}
          disabled={personas.length <= 1}
          title="Previous"
        >
          ←
        </button>
        
        <div className="gallery-counter">
          {currentIndex + 1} of {personas.length}
        </div>
        
        <button 
          className="gallery-nav-btn next-btn" 
          onClick={handleNext}
          disabled={personas.length <= 1}
          title="Next"
        >
          →
        </button>
      </div>

      {discussionGroupMembers.size > 0 && (
        <div className="discussion-group-summary">
          <h4>Discussion Group Members ({discussionGroupMembers.size})</h4>
          <div className="group-members-list">
            {Array.from(discussionGroupMembers).map(memberId => {
              const member = personas.find(p => 
                (p.id || `${p.first_name}-${p.last_name}`) === memberId
              );
              return member ? (
                <span key={memberId} className="group-member-chip">
                  {member.first_name} {member.last_name}
                </span>
              ) : null;
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default PersonaGallery;
