// Debate data parser and simulator
export const parseDebateData = (conversationText) => {
  // This function is no longer needed with the new flat structure
  // but kept for backward compatibility
  return null;
};

// Simulate debate with the new flat message structure
export const simulateDebate = async (debateData, onMessage, onSystemMessage, delay = 2000) => {
  const messages = [];
  
  // Process each message in order
  for (const messageData of debateData.messages) {
    if (messageData.type === 'system') {
      // System messages (round announcements, voting results, etc.)
      const systemMessage = {
        id: Date.now() + Math.random(),
        type: 'system',
        text: messageData.text,
        timestamp: new Date().toISOString()
      };
      
      await new Promise(resolve => setTimeout(resolve, delay));
      onSystemMessage(systemMessage);
      messages.push(systemMessage);
      
      // Additional delay after system message before first person message
      await new Promise(resolve => setTimeout(resolve, 2000));
      
    } else if (messageData.type === 'vote' || messageData.type === 'discuss') {
      // Participant messages (voting or discussion)
      const message = {
        id: Date.now() + Math.random(),
        participantId: messageData.agent_name.split(' ')[0].toLowerCase(),
        text: messageData.text,
        timestamp: new Date().toISOString(),
        type: 'agent',
        hate_speech: messageData.hate_speech,
        extremism: messageData.extremism
      };
      
      // Wait for typing animation to complete (8 seconds) + 2 seconds delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      await onMessage(message);
      messages.push(message);
    }
  }

  return messages;
};