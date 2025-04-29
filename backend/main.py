from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel, EmailStr, validator
import jwt
from jose import JWTError
import os
from dotenv import load_dotenv
import cv2
import ai_models.exercise_detector as exercise_detector
import numpy as np
import re
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
from pymongo import WriteConcern

from database import (
    users, exercises, workouts, user_profiles,
    exercise_plans, ai_models
)
from auth import (
    Token, UserCreate, User, authenticate_user,
    create_access_token, get_password_hash,
    ACCESS_TOKEN_EXPIRE_MINUTES, get_user
)
from ai_models.exercise_detector import ExerciseDetector


load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"

app = FastAPI(
    title="AI Fitness Tracker API",
    description="Backend API for AI Fitness Tracker",
    version="1.0.0"
)

# Configure CORS
origins = [
    "http://localhost:8501",  # Streamlit default port
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom startup event
@app.on_event("startup")
async def startup_event():
    print("\n=== Server Starting ===")
    print(f"CORS Origins configured: {origins}")
    print("MongoDB connection will be tested on first request")
    print("Server is ready to accept connections")

# Add custom shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    print("\n=== Server Shutting Down ===")

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class ExerciseData(BaseModel):
    username: str
    exercise_type: str
    duration: int
    calories_burned: float
    form_score: float
    feedback: List[str]
    reps: int
    incorrect_reps: int
    timestamp: datetime
    rep_history: List[dict] = []  # New: per-rep analytics

class WorkoutPlan(BaseModel):
    username: str
    exercises: List[dict]
    duration: int
    difficulty: str
    created_at: datetime

class Achievement(BaseModel):
    title: str
    description: str
    date: str  # ISO format

class UserProfile(BaseModel):
    username: str
    full_name: str
    age: int
    weight: float
    height: float
    fitness_goal: str
    fitness_level: str
    weekly_goal: int
    preferred_exercises: list
    target_weight: float
    workout_duration: int
    notes: str
    achievements: List[Achievement] = []  # New field for multiple objects

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str

    @validator('username')
    def username_alphanumeric(cls, v):
        if not re.match("^[a-zA-Z0-9_-]+$", v):
            raise ValueError('Username must be alphanumeric')
        return v

    @validator('password')
    def password_min_length(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        return v

# Authentication
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(username=username)
    if user is None:
        raise credentials_exception
    return user

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        if isinstance(e, ServerSelectionTimeoutError):
            return JSONResponse(
                status_code=503,
                content={"detail": "Database connection timeout. Please try again."}
            )
        raise e

# Routes
@app.post("/register", response_model=User)
async def register_user(user: UserCreate):
    try:
        print(f"\n=== Registration Request ===")
        print(f"Username: {user.username}")
        print(f"Email: {user.email}")
        print(f"Full Name: {user.full_name}")
        
        try:
            # Check if user exists with timeout
            existing_user = users.find_one(
                {"$or": [
                    {"username": user.username},
                    {"email": user.email}
                ]},
                max_time_ms=5000  # 5 second timeout
            )
            
            if existing_user:
                if existing_user["username"] == user.username:
                    print(f"❌ Registration failed: Username {user.username} already exists")
                    raise HTTPException(
                        status_code=400,
                        detail="Username already registered"
                    )
                else:
                    print(f"❌ Registration failed: Email {user.email} already exists")
                    raise HTTPException(
                        status_code=400,
                        detail="Email already registered"
                    )
            
            # Create user data
            hashed_password = get_password_hash(user.password)
            user_data = {
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "hashed_password": hashed_password,
                "disabled": False,
                "created_at": datetime.utcnow()
            }
            
            print(f"Attempting to insert new user into database...")
            
            # Insert user with write concern
            write_concern = WriteConcern(w=1, wtimeout=5000)
            result = users.with_options(write_concern=write_concern).insert_one(user_data)
            
            if not result.inserted_id:
                print("❌ Database insert failed: No inserted_id returned")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create user: Database error"
                )
            
            print(f"✅ User created successfully with ID: {result.inserted_id}")
            
            # Verify the user was actually created
            created_user = users.find_one(
                {"_id": result.inserted_id},
                max_time_ms=5000  # 5 second timeout
            )
            
            if not created_user:
                print("❌ User verification failed: Could not find created user")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to verify user creation"
                )
            
            print(f"✅ User verified in database")
            
            return {
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "disabled": False
            }
            
        except ServerSelectionTimeoutError:
            print("❌ Database timeout error")
            raise HTTPException(
                status_code=503,
                detail="Database connection timeout. Please try again."
            )
        except OperationFailure as e:
            print(f"❌ Database operation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database operation failed: {str(e)}"
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Unexpected error during registration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/profile", response_model=UserProfile)
async def create_user_profile(profile: UserProfile, current_user: User = Depends(get_current_user)):
    if current_user.username != profile.username:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to create profile for other users"
        )
    
    profile_data = profile.dict()
    result = user_profiles.update_one(
        {"username": profile.username},
        {"$set": profile_data},
        upsert=True
    )
    
    if not result.acknowledged:
        raise HTTPException(status_code=400, detail="Failed to create/update profile")
    
    return profile

@app.get("/profile/{username}", response_model=UserProfile)
async def get_user_profile(username: str, current_user: User = Depends(get_current_user)):
    if current_user.username != username:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to view other users' profiles"
        )
    
    profile = user_profiles.find_one({"username": username})
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return UserProfile(**profile)

@app.post("/exercise/analyze")
async def analyze_exercise(
    exercise_type: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    # Read and process the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Initialize exercise detector
    detector = ExerciseDetector()
    
    # Detect pose and analyze exercise
    pose_data = detector.detect_pose(frame)
    if not pose_data:
        raise HTTPException(status_code=400, detail="No person detected in the image")
    
    analysis = detector.analyze_exercise(exercise_type, pose_data["landmarks"])
    
    return analysis

@app.post("/exercise/")
async def save_exercise(data: ExerciseData, current_user: User = Depends(get_current_user)):
    """Save exercise data with proper validation and error handling."""
    try:
        if current_user.username != data.username:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to save exercise data for other users"
            )
        
        exercise_data = {
            "username": data.username,
            "exercise_type": data.exercise_type,
            "duration": data.duration,
            "calories_burned": data.calories_burned,
            "form_score": data.form_score,
            "feedback": data.feedback,
            "reps": data.reps,
            "incorrect_reps": data.incorrect_reps,
            "timestamp": data.timestamp,
            "rep_history": data.rep_history
        }
        
        # Use write concern for data persistence
        write_concern = WriteConcern(w=1, wtimeout=5000)
        result = exercises.with_options(write_concern=write_concern).insert_one(exercise_data)
        
        if not result.inserted_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to save exercise data: Database error"
            )
        
        print(f"✅ Exercise data saved successfully for user: {data.username}")
        return {"message": "Exercise data saved successfully!", "id": str(result.inserted_id)}
        
    except Exception as e:
        print(f"❌ Error saving exercise data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save exercise data: {str(e)}"
        )

@app.get("/exercise/{username}", response_model=List[dict])
async def get_exercises(username: str, current_user: User = Depends(get_current_user)):
    if current_user.username != username:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to view other users' exercise data"
        )
    
    user_exercises = list(exercises.find({"username": username}))
    if not user_exercises:
        return []
    
    # Convert ObjectId to string for JSON serialization
    for exercise in user_exercises:
        exercise["_id"] = str(exercise["_id"])
        exercise["timestamp"] = exercise["timestamp"].isoformat()
    
    return user_exercises

@app.get("/exercise/stats/{username}")
async def get_exercise_stats(username: str, current_user: User = Depends(get_current_user)):
    if current_user.username != username:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to view other users' exercise stats"
        )
    
    pipeline = [
        {"$match": {"username": username}},
        {"$group": {
            "_id": "$exercise_type",
            "total_count": {"$sum": 1},
            "total_duration": {"$sum": "$duration"},
            "total_calories": {"$sum": "$calories_burned"},
            "avg_form_score": {"$avg": "$form_score"},
            "last_workout": {"$max": "$timestamp"}
        }}
    ]
    
    stats = list(exercises.aggregate(pipeline))
    return stats

@app.post("/workout/plan")
async def create_workout_plan(plan: WorkoutPlan, current_user: User = Depends(get_current_user)):
    if current_user.username != plan.username:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to create workout plans for other users"
        )
    
    plan_data = plan.dict()
    plan_data["created_at"] = datetime.utcnow()
    
    result = exercise_plans.insert_one(plan_data)
    if not result.inserted_id:
        raise HTTPException(status_code=400, detail="Failed to create workout plan")
    
    return {"message": "Workout plan created successfully!"}

@app.get("/workout/plan/{username}")
async def get_workout_plans(username: str, current_user: User = Depends(get_current_user)):
    if current_user.username != username:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to view other users' workout plans"
        )
    
    plans = list(exercise_plans.find({"username": username}))
    if not plans:
        return []
    
    # Convert ObjectId to string for JSON serialization
    for plan in plans:
        plan["_id"] = str(plan["_id"])
        plan["created_at"] = plan["created_at"].isoformat()
    
    return plans

@app.get("/workouts/history")
async def get_workout_history(
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 10
):
    """Retrieve workout history for the current user with pagination support."""
    try:
        print(f"\n=== Fetching Workout History ===")
        print(f"User: {current_user['username']}")
        print(f"Skip: {skip}, Limit: {limit}")
        
        # Get total count for pagination
        total_workouts = workouts.count_documents({"user_id": current_user["_id"]})
        
        # Get paginated workouts for the user
        user_workouts = list(workouts.find(
            {"user_id": current_user["_id"]},
            {"_id": 0, "date": 1, "exercises": 1}  # Only return date and exercises
        ).sort("date", -1).skip(skip).limit(limit))
        
        # Convert dates to ISO format for JSON serialization
        for workout in user_workouts:
            workout["date"] = workout["date"].isoformat()
            
        print(f"Found {len(user_workouts)} workouts")
            
        return {
            "workouts": user_workouts,
            "total": total_workouts,
            "has_more": (skip + limit) < total_workouts
        }
        
    except Exception as e:
        print(f"❌ Error retrieving workout history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving workout history: {str(e)}"
        ) 