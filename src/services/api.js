import axios from 'axios';

const API_BASE_URL = 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes timeout for audio processing
});

export const uploadAudioFile = async (file) => {
  const formData = new FormData();
  formData.append('audio_file', file);

  try {
    const response = await api.post('/process-audio', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      // Server responded with error status
      throw new Error(error.response.data.detail || 'Server error occurred');
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('Unable to connect to server. Please make sure the backend is running.');
    } else {
      // Something else happened
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
};

export const getHealthStatus = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw new Error('Backend server is not available');
  }
};

export const runJuryDebate = async (statement, rounds = 3) => {
  try {
    const response = await api.post('/jury-debate', {
      statement: statement,
      rounds: rounds
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Jury debate failed');
    } else if (error.request) {
      throw new Error('Unable to connect to server. Please make sure the backend is running.');
    } else {
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
};

export const createJuryDebateWebSocket = () => {
  const wsUrl = API_BASE_URL.replace('http', 'ws') + '/ws/jury-debate';
  const ws = new WebSocket(wsUrl);
  
  return ws;
};
