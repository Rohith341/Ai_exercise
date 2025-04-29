# AI Fitness Tracker

A full-stack fitness tracking application with AI-powered exercise form analysis.

## Features

- User authentication and profile management
- Exercise tracking with AI form analysis
- Workout plan creation and management
- Real-time exercise statistics and visualizations
- AI-powered exercise form feedback

## Tech Stack

- Backend: FastAPI
- Frontend: Streamlit
- Database: MongoDB
- AI/ML: TensorFlow, MediaPipe, OpenCV
- Authentication: JWT

## Prerequisites

- Python 3.8+
- MongoDB
- Webcam (for exercise form analysis)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai_fitness_tracker
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Copy `.env.example` to `.env`
- Update the values in `.env` with your configuration

5. Start MongoDB:
```bash
mongod
```

## Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. Start the frontend application:
```bash
cd frontend
streamlit run app.py
```

3. Access the application:
- Backend API: http://localhost:8000
- Frontend: http://localhost:8501

## Usage

1. Register a new account or login
2. Complete your profile information
3. Use the Exercise Tracker to:
   - Record exercises with form analysis
   - View exercise statistics
   - Create and manage workout plans
4. Use the webcam to get real-time form feedback during exercises

## API Endpoints

- POST /register - Register a new user
- POST /token - Login and get access token
- POST /profile - Create/update user profile
- GET /profile/{username} - Get user profile
- POST /exercise/analyze - Analyze exercise form
- POST /exercise/ - Save exercise data
- GET /exercise/{username} - Get user's exercises
- GET /exercise/stats/{username} - Get exercise statistics
- POST /workout/plan - Create workout plan
- GET /workout/plan/{username} - Get user's workout plans

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. "# mini_project" 
"# Ai_exercise" 
"# Minor-Project" 
