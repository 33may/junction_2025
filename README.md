# Audio Processing App

A React frontend application with FastAPI backend for processing audio files and detecting flagged segments.

## Features

### Audio Processing
- **Drag & Drop Audio Upload**: Easy file upload with visual feedback
- **Audio Analysis**: Detects various audio issues including:
  - Silence segments
  - Loud segments
  - Frequency anomalies
  - Clipping/distortion
- **Real-time Processing**: Shows processing status and results
- **Detailed Results**: Displays flagged segments with timestamps, severity, and descriptions

### AI Jury Debate System
- **Multi-Agent Debate**: AI personas with diverse political views debate audio content
- **Real-time WebSocket Communication**: Live debate with typing indicators and immediate responses
- **Persona Diversity**: Generated personas represent different political ideologies, backgrounds, and viewpoints
- **Voting System**: Each agent votes on hate speech and extremism classifications
- **Discussion Rounds**: Agents can discuss and challenge each other's positions
- **Final Verdict**: Majority-based decision on content classification

### Persona Generation
- **AI-Generated Personas**: Create diverse debate participants using OpenAI
- **Realistic Demographics**: Personas include age, political views, education, background stories
- **Balanced Representation**: Ensures diverse political spectrum from far-left to far-right
- **Customizable**: Generate different sets of personas for varied debate scenarios

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Install Node.js dependencies
   npm install
   ```

## Running the Application

### Option 1: Use the startup script (Recommended)
```bash
./start.sh
```

### Option 2: Manual startup

1. Start the FastAPI backend:
   ```bash
   cd backend
   python main.py
   ```~

2. In a new terminal, start the React frontend:
   ```bash
   npm start
   ```

3. Open your browser and go to `http://localhost:3000`

## API Endpoints

### Audio Processing
- `GET /health` - Health check endpoint
- `POST /process-audio` - Process uploaded audio file

### Jury Debate System
- `WebSocket /ws/jury-debate` - Real-time debate communication
  - Sends system messages, typing indicators, agent votes, and discussion
  - Uses transcribed audio content as debate statement
  - Returns final verdict with voting results

## Supported Audio Formats

- MP3
- WAV
- M4A
- OGG
- FLAC

## File Size Limit

Maximum file size: 50MB

## How It Works

### Audio Processing Workflow
1. **Upload**: Drag and drop an audio file onto the upload area
2. **Processing**: The file is sent to the FastAPI backend for analysis
3. **Analysis**: The backend uses librosa to analyze the audio for various issues:
   - Silence detection using RMS energy
   - Loud segment detection
   - Frequency anomaly detection using spectral centroid
   - Clipping detection by analyzing amplitude peaks
4. **Results**: Flagged segments are displayed with timestamps, severity levels, and descriptions

### Jury Debate Workflow
1. **Audio Transcription**: Uploaded audio is transcribed using Whisper
2. **Persona Selection**: AI personas with diverse political views are loaded
3. **Debate Initiation**: WebSocket connection starts the debate process
4. **Voting Rounds**: Each agent analyzes the transcribed content and votes on:
   - Hate speech classification (true/false)
   - Extremism classification (true/false)
5. **Discussion Phase**: Agents discuss their reasoning and challenge others' positions
6. **Final Verdict**: Majority vote determines the final classification
7. **Results Display**: Final decision and voting breakdown are shown

### Persona Generation Process
1. **AI Generation**: Uses OpenAI to create diverse personas with realistic demographics
2. **Political Diversity**: Ensures representation across the political spectrum
3. **Background Stories**: Each persona has a detailed backstory explaining their worldview
4. **Validation**: Generated personas are validated against predefined schemas
5. **Storage**: Personas are saved to JSON files for reuse in debates

## Project Structure

```
junction_2025/
├── src/
│   ├── components/
│   │   ├── AudioUploader.js          # Audio file upload component
│   │   ├── AudioUploader.css
│   │   ├── ProcessingResults.js      # Audio analysis results display
│   │   ├── ProcessingResults.css
│   │   ├── DebateChat.js             # Real-time debate chat interface
│   │   ├── DebateChat.css
│   │   ├── DebateResults.js          # Final debate verdict display
│   │   ├── DebateResults.css
│   │   ├── PersonaGallery.js         # Persona selection interface
│   │   ├── PersonaGallery.css
│   │   ├── ThemeToggle.js            # Dark/light theme toggle
│   │   └── ThemeToggle.css
│   ├── services/
│   │   └── api.js                    # API communication utilities
│   ├── utils/
│   │   └── debateSimulator.js        # Mock debate simulation
│   ├── images/                       # Persona profile images
│   ├── App.js                        # Main application component
│   ├── App.css
│   ├── index.js
│   └── index.css
├── backend/
│   ├── main.py                       # FastAPI server with audio processing & WebSocket
│   ├── jury.py                       # AI jury debate logic
│   ├── generate_personas.py          # Persona generation using OpenAI
│   ├── audio_pipeline.py             # Audio processing pipeline
│   ├── diar_pipeline.py              # Speaker diarization
│   └── adsr_pipeline.py              # Audio analysis pipeline
├── data/
│   ├── personas.json                 # Generated persona data
│   └── *.mp3                         # Sample audio files
├── public/
│   └── index.html
├── package.json
├── requirements.txt
└── start.sh
```

## Development

### Audio Analysis
To modify the audio analysis algorithms, edit the functions in `backend/main.py`:
- `detect_silence()` - Silence detection
- `detect_loud_segments()` - Loud segment detection
- `detect_frequency_anomalies()` - Frequency analysis
- `detect_clipping()` - Clipping detection

### Jury Debate System
To modify the debate logic, edit the functions in `backend/jury.py`:
- `_load_personalities()` - Load persona data
- `_agent_step()` - Individual agent voting logic
- `_build_discussion_prompt()` - Discussion phase prompts
- `_get_discussion_response()` - Agent discussion responses

### Persona Generation
To generate new personas, run:
```bash
cd backend
python generate_personas.py
```

This will:
- Generate 9 diverse personas using OpenAI
- Save them to `data/personas.json`
- Create realistic demographics and backstories

### WebSocket Communication
The debate system uses WebSocket messages:
- `system_message` - System announcements
- `agent_typing_start/stop` - Typing indicators
- `agent_vote` - Agent voting results
- `agent_discussion` - Discussion responses
- `debate_completed` - Final results

## Troubleshooting

### General Issues
- **Backend not starting**: Make sure Python dependencies are installed
- **Frontend not loading**: Check if Node.js dependencies are installed
- **CORS errors**: Ensure the backend is running on port 8001
- **File upload fails**: Check file size (max 50MB) and format

### Jury Debate Issues
- **No personas available**: Run `python generate_personas.py` to create personas
- **WebSocket connection fails**: Check if backend is running and OpenAI API key is set
- **Debate not starting**: Ensure audio has been processed first (transcription required)
- **Typing indicators not showing**: Check browser console for WebSocket message errors

### Persona Generation Issues
- **OpenAI API errors**: Ensure `OPENAI_KEY` environment variable is set
- **Persona generation fails**: Check internet connection and API quota
- **Invalid persona data**: Check that generated JSON matches the expected schema

### Environment Variables
Required environment variables:
- `OPENAI_KEY` or `OPENAI_API_KEY` - OpenAI API key for persona generation and jury debate
- `JURY_OPENAI_MODEL` - Optional, defaults to "gpt-4o-mini"

## Jury Debate System Details

### How the AI Jury Works
The jury system uses multiple AI personas with diverse political views to analyze audio content:

1. **Persona Diversity**: Each persona define with the schema in `generate_personas.py`


2. **Voting Process**: Each agent votes on two classifications:
   - **Hate Speech**: Whether the content contains hateful language
   - **Extremism**: Whether the content promotes extremist views

3. **Discussion Rounds**: Agents can challenge each other's positions and provide reasoning

4. **Final Decision**: Majority vote determines the final classification

### Persona Generation Details

The persona generation system creates realistic AI debate participants:

### WebSocket Message Types

The debate system communicates through specific message types:

- **`system_message`**: Round announcements, voting results
- **`agent_typing_start`**: Shows which agent is thinking
- **`agent_typing_stop`**: Hides typing indicator
- **`agent_vote`**: Agent's vote with reasoning
- **`agent_discussion`**: Agent's discussion response
- **`debate_completed`**: Final results and verdict

### Customization Options

#### Modifying Debate Logic
Edit `backend/jury.py` to customize:
- Voting criteria and thresholds
- Discussion prompts and responses
- Agent personality instructions
- Debate round structure


## License

This project is for educational purposes.
