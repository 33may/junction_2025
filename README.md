# Audio Processing App

A React frontend application with FastAPI backend for processing audio files and detecting flagged segments.

## Features

- **Drag & Drop Audio Upload**: Easy file upload with visual feedback
- **Audio Analysis**: Detects various audio issues including:
  - Silence segments
  - Loud segments
  - Frequency anomalies
  - Clipping/distortion
- **Real-time Processing**: Shows processing status and results
- **Detailed Results**: Displays flagged segments with timestamps, severity, and descriptions

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

- `GET /health` - Health check endpoint
- `POST /process-audio` - Process uploaded audio file

## Supported Audio Formats

- MP3
- WAV
- M4A
- OGG
- FLAC

## File Size Limit

Maximum file size: 50MB

## How It Works

1. **Upload**: Drag and drop an audio file onto the upload area
2. **Processing**: The file is sent to the FastAPI backend for analysis
3. **Analysis**: The backend uses librosa to analyze the audio for various issues:
   - Silence detection using RMS energy
   - Loud segment detection
   - Frequency anomaly detection using spectral centroid
   - Clipping detection by analyzing amplitude peaks
4. **Results**: Flagged segments are displayed with timestamps, severity levels, and descriptions

## Project Structure

```
junction_2025/
├── src/
│   ├── components/
│   │   ├── AudioUploader.js
│   │   ├── AudioUploader.css
│   │   ├── ProcessingResults.js
│   │   └── ProcessingResults.css
│   ├── services/
│   │   └── api.js
│   ├── App.js
│   ├── App.css
│   ├── index.js
│   └── index.css
├── backend/
│   └── main.py
├── public/
│   └── index.html
├── package.json
├── requirements.txt
└── start.sh
```

## Development

To modify the audio analysis algorithms, edit the functions in `backend/main.py`:
- `detect_silence()` - Silence detection
- `detect_loud_segments()` - Loud segment detection
- `detect_frequency_anomalies()` - Frequency analysis
- `detect_clipping()` - Clipping detection

## Troubleshooting

- **Backend not starting**: Make sure Python dependencies are installed
- **Frontend not loading**: Check if Node.js dependencies are installed
- **CORS errors**: Ensure the backend is running on port 8000
- **File upload fails**: Check file size (max 50MB) and format

## License

This project is for educational purposes.
