import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import cv2
import numpy as np
from PIL import Image
import io
import re
import time

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.ai_models.exercise_detector import ExerciseDetector

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"  # Using IP instead of localhost
EXERCISE_TYPES = [
    "Push-ups",
    "Squats",
    "Lunges",
    "Plank",
    "Bicep Curls",
    "Shoulder Press",
    "Deadlifts",
    "Jumping Jacks"
]

# Calorie calculation constants (calories per minute)
EXERCISE_CALORIES = {
    "Push-ups": 7,
    "Squats": 5,
    "Lunges": 6,
    "Plank": 3,
    "Bicep Curls": 4,
    "Shoulder Press": 5,
    "Deadlifts": 8,
    "Jumping Jacks": 10
}

# Configure retry strategy
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=0.5,  # wait 0.5, 1, 2 seconds between retries
    status_forcelist=[500, 502, 503, 504],  # HTTP status codes to retry on
    allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]  # Allow retrying on POST
)

# Create session with retry strategy
http = requests.Session()
http.mount("http://", HTTPAdapter(max_retries=retry_strategy))
http.mount("https://", HTTPAdapter(max_retries=retry_strategy))

def check_server_health():
    """Check if the backend server is running and healthy"""
    try:
        response = http.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

def wait_for_server(timeout=30):
    """Wait for server to become available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_server_health():
            return True
        time.sleep(1)
    return False

# Check server health before starting
if not check_server_health():
    st.error("⚠️ Backend server is not running. Please start the backend server first.")
    st.info("To start the backend server, run:\n```\ncd backend\npython -m uvicorn main:app --host 127.0.0.1 --port 8000\n```")
    st.stop()

def get_auth_headers():
    """Get authentication headers with the current access token"""
    if 'access_token' in st.session_state and st.session_state.access_token:
        return {
            "Authorization": f"Bearer {st.session_state.access_token}",
            "Content-Type": "application/json"
        }
    return {"Content-Type": "application/json"}

def show_home_page():
    st.title("🏋️‍♂️ Welcome to AI Fitness Tracker")
    
    # Fetch user profile and workout history
    profile_data = get_profile_data()
    workout_history = get_workout_history()
    
    # Fallbacks if data is missing
    if not profile_data:
        profile_data = {}
    if not workout_history or not isinstance(workout_history, list):
        workout_history = []

    # Get today's workouts
    today = datetime.now().date()
    todays_workouts = [
        w for w in workout_history 
        if isinstance(w, dict) and 'date' in w and datetime.strptime(w['date'], '%Y-%m-%d').date() == today
    ]
    
    # Calculate metrics
    num_todays_workouts = len(todays_workouts)
    num_unique_exercises_today = len(set(w.get('exercise_type', '') for w in todays_workouts))

    # Calculate active streak
    streak = calculate_streak(workout_history) if workout_history else 0

    # Calculate fitness score (average form score of last 5 workouts, or 0)
    # Ensure we're working with a list and handle slicing safely
    valid_workouts = [w for w in workout_history if isinstance(w, dict)]
    recent_workouts = valid_workouts[-5:] if valid_workouts else []
    recent_scores = [w.get('form_score', 0) for w in recent_workouts if w.get('form_score') is not None]
    fitness_score = int(sum(recent_scores) / len(recent_scores)) if recent_scores else 0

    # Dashboard layout with columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Today's Workouts", value=str(num_todays_workouts))
        if st.button("➕ Start New Workout"):
            st.session_state.current_page = "Exercise Tracker"
            st.rerun()
            
    with col2:
        st.metric(label="Unique Exercises Today", value=str(num_unique_exercises_today))
    
    with col3:
        st.metric(label="Active Streak", value=f"{streak} days")
        if st.button("📊 View Statistics"):
            st.session_state.current_page = "Profile"
            st.rerun()
            
    with col4:
        st.metric(label="Fitness Score", value=str(fitness_score))
        if st.button("👤 Update Profile"):
            st.session_state.current_page = "Profile"
            st.rerun()
    
    # Quick actions section
    st.subheader("Quick Actions")
    quick_action = st.selectbox(
        "What would you like to do?",
        [
            "Select an action...",
            "Start a new workout",
            "Create a workout plan",
            "View exercise history",
            "Update profile"
        ]
    )
    
    if quick_action != "Select an action...":
        if quick_action == "Start a new workout":
            st.session_state.current_page = "Exercise Tracker"
        elif quick_action == "Create a workout plan":
            st.session_state.current_page = "Workout Plans"
        elif quick_action == "View exercise history":
            st.session_state.current_page = "Profile"
        elif quick_action == "Update profile":
            st.session_state.current_page = "Profile"
        st.rerun()

    # Today's Exercise History as Cards
    st.markdown("---")
    st.subheader("Today's Exercise History")
    if todays_workouts:
        for w in todays_workouts:
            with st.container():
                st.markdown(f"### {w.get('exercise_type', 'Exercise')}")
                cols = st.columns(4)
                cols[0].metric("Duration", f"{w.get('duration', 0)} min")
                cols[1].metric("Form Score", f"{w.get('form_score', 0)}%")
                cols[2].metric("Calories", f"{w.get('calories_burned', 0)}")
                cols[3].metric("Reps", f"{w.get('reps', 0)} | ❌ {w.get('incorrect_reps', 0)}")
                if w.get('feedback'):
                    st.write("**Feedback:**")
                    for fb in w['feedback']:
                        st.write(f"- {fb}")
                st.markdown("---")
    else:
        st.info("No exercises performed today. Start a new workout to see your progress here!")

def register():
    st.title("📝 Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        full_name = st.text_input("Full Name")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        submit_button = st.form_submit_button("Register")
        
        if submit_button:
            st.info("Processing registration...")
            
            # Validate required fields
            if not all([username, email, password]):
                st.error("Username, email, and password are required!")
                return
            
            # Validate password match
            if password != confirm_password:
                st.error("Passwords do not match!")
                return
            
            # Validate password length
            if len(password) < 6:
                st.error("Password must be at least 6 characters long!")
                return
            
            # Validate username format
            if not re.match("^[a-zA-Z0-9_-]+$", username):
                st.error("Username must contain only letters, numbers, underscores, and hyphens!")
                return
            
            # Validate email format
            if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
                st.error("Please enter a valid email address!")
                return
            
            try:
                # Print request data for debugging
                print("\n=== Registration Request ===")
                print(f"API Endpoint: {API_BASE_URL}/register")
                print(f"Request data: username={username}, email={email}, full_name={full_name}")
                
                # Prepare request data
                request_data = {
                    "username": username,
                    "email": email,
                    "full_name": full_name if full_name else None,  # Make sure to handle empty full_name
                    "password": password
                }
                
                # Make the request with proper headers and retry strategy
                response = http.post(
                    f"{API_BASE_URL}/register",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=30  # Increased timeout
                )
                
                # Print response details for debugging
                print("\n=== Registration Response ===")
                print(f"Status code: {response.status_code}")
                print(f"Response headers: {response.headers}")
                print(f"Response content: {response.text}")
                
                if response.status_code == 200:
                    st.success("✅ Registration successful! Redirecting to login...")
                    st.session_state.show_register = False
                    st.session_state.show_login = True
                    time.sleep(2)  # Show success message for 2 seconds
                    st.rerun()
                elif response.status_code == 503:
                    st.error("❌ Server is temporarily unavailable. Please try again in a few moments.")
                    print("\n❌ Server temporarily unavailable (503)")
                else:
                    try:
                        error_data = response.json()
                        error_detail = error_data.get("detail", "Registration failed")
                        st.error(f"❌ Registration failed: {error_detail}")
                    except:
                        error_detail = response.text
                        st.error(f"❌ Registration failed: {error_detail}")
                    print(f"❌ Registration failed with status {response.status_code}: {error_detail}")
            except requests.exceptions.ConnectionError:
                error_msg = f"❌ Could not connect to the server at {API_BASE_URL}. Please make sure the backend server is running."
                st.error(error_msg)
                print(f"\n{error_msg}")
            except requests.exceptions.Timeout:
                error_msg = "❌ Request timed out. The server might be busy, please try again."
                st.error(error_msg)
                print(f"\n{error_msg}")
            except Exception as e:
                error_msg = f"❌ Registration failed: {str(e)}"
                st.error(error_msg)
                print(f"\n{error_msg}")

def login():
    if 'access_token' in st.session_state and st.session_state.access_token:
        return True
    
    st.title("🔐 Login")
    
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    
    if st.session_state.show_register:
        register()
        if st.button("Back to Login"):
            st.session_state.show_register = False
            st.rerun()
        return False
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password!")
                return False
            
            try:
                response = requests.post(
                    f"{API_BASE_URL}/token",
                    data={
                        "username": username,
                        "password": password
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    st.session_state.access_token = token_data["access_token"]
                    st.session_state.username = username
                    st.success("✅ Login successful! Redirecting to home...")
                    time.sleep(1)  # Show success message for 1 second
                    st.session_state.current_page = "Home"
                    st.rerun()
                    return True
                else:
                    error_detail = response.json().get("detail", "Invalid username or password")
                    st.error(f"Login failed: {error_detail}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the server. Please try again later.")
            except Exception as e:
                st.error(f"Login failed: {str(e)}")
    
    if st.button("Create New Account"):
        st.session_state.show_register = True
        st.rerun()
    
    return False

def get_profile_data():
    """Retrieve user profile data from the backend."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/profile/{st.session_state.username}",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get profile data: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error retrieving profile data: {str(e)}")
        return None

def update_profile(profile_data):
    """Update user profile data in the backend."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/profile",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            json=profile_data,
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error updating profile: {str(e)}")
        return False

def show_profile_page():
    st.title("👤 Profile")
    try:
        # Get user profile
        response = requests.get(
            f"{API_BASE_URL}/profile/{st.session_state.username}",
            headers=get_auth_headers(),
            timeout=10
        )
        # Initialize default profile if not exists
        if response.status_code == 404:
            default_profile = {
                "username": st.session_state.username,
                "full_name": st.session_state.get("full_name", "Your Name"),
                "age": 25,
                "gender": "Other",
                "weight": 70.0,
                "height": 170.0,
                "fitness_goal": "General Fitness",
                "fitness_level": "Beginner",
                "weekly_goal": 3,
                "preferred_exercises": ["Push-ups", "Squats", "Lunges"],
                "target_weight": 70.0,
                "workout_duration": 30,
                "notes": "",
                "achievements": []
            }
            create_response = requests.post(
                f"{API_BASE_URL}/profile",
                json=default_profile,
                headers=get_auth_headers(),
                timeout=10
            )
            if create_response.status_code == 200:
                profile = default_profile
            else:
                st.error("Failed to create profile")
                return
        elif response.status_code == 200:
            profile = response.json()
        else:
            st.error(f"Failed to load profile: {response.text}")
            return

        # Initialize session state for all fields if not present
        profile_fields = {
            "full_name": profile.get("full_name", ""),
            "age": profile.get("age", 25),
            "gender": profile.get("gender", "Other"),
            "weight": profile.get("weight", 70.0),
            "height": profile.get("height", 170.0),
            "fitness_goal": profile.get("fitness_goal", "General Fitness"),
            "fitness_level": profile.get("fitness_level", "Beginner"),
            "weekly_goal": profile.get("weekly_goal", 3),
            "preferred_exercises": profile.get("preferred_exercises", ["Push-ups", "Squats", "Lunges"]),
            "target_weight": profile.get("target_weight", profile.get("weight", 70.0)),
            "workout_duration": profile.get("workout_duration", 30),
            "notes": profile.get("notes", ""),
            "achievements": profile.get("achievements", [])
        }
        for k, v in profile_fields.items():
            if f"profile_{k}" not in st.session_state:
                st.session_state[f"profile_{k}"] = v

        # Achievements management in session state
        if "profile_achievements" not in st.session_state:
            st.session_state["profile_achievements"] = profile_fields["achievements"]

        def collect_profile_data():
            return {
                "username": st.session_state.username,
                "full_name": st.session_state["profile_full_name"],
                "age": st.session_state["profile_age"],
                "gender": st.session_state["profile_gender"],
                "weight": st.session_state["profile_weight"],
                "height": st.session_state["profile_height"],
                "fitness_goal": st.session_state["profile_fitness_goal"],
                "fitness_level": st.session_state["profile_fitness_level"],
                "weekly_goal": st.session_state["profile_weekly_goal"],
                "preferred_exercises": st.session_state["profile_preferred_exercises"],
                "target_weight": st.session_state["profile_target_weight"],
                "workout_duration": st.session_state["profile_workout_duration"],
                "notes": st.session_state["profile_notes"],
                "achievements": st.session_state["profile_achievements"]
            }

        # Wrap all tabs in a single form
        with st.form("profile_full_form"):
            tab1, tab2, tab3 = st.tabs(["Basic Info", "Fitness Goals", "Achievements"])

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state["profile_full_name"] = st.text_input("Full Name", value=st.session_state["profile_full_name"])
                    st.session_state["profile_age"] = st.number_input("Age", min_value=1, max_value=120, value=st.session_state["profile_age"])
                    st.session_state["profile_gender"] = st.selectbox(
                        "Gender",
                        ["Male", "Female", "Other"],
                        index=["Male", "Female", "Other"].index(st.session_state["profile_gender"])
                    )
                with col2:
                    st.session_state["profile_weight"] = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=st.session_state["profile_weight"])
                    st.session_state["profile_height"] = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=st.session_state["profile_height"])
                    if st.session_state["profile_height"] and st.session_state["profile_weight"]:
                        bmi = st.session_state["profile_weight"] / ((st.session_state["profile_height"]/100) ** 2)
                        st.metric("BMI", f"{bmi:.1f}")
                        if bmi < 18.5:
                            st.warning("Underweight")
                        elif bmi < 25:
                            st.success("Normal weight")
                        elif bmi < 30:
                            st.warning("Overweight")
                        else:
                            st.error("Obese")
                st.session_state["profile_fitness_goal"] = st.selectbox(
                    "Primary Goal",
                    ["Weight Loss", "Muscle Gain", "Endurance", "Flexibility", "General Fitness"],
                    index=["Weight Loss", "Muscle Gain", "Endurance", "Flexibility", "General Fitness"].index(st.session_state["profile_fitness_goal"])
                )
                st.session_state["profile_fitness_level"] = st.selectbox(
                    "Current Fitness Level",
                    ["Beginner", "Intermediate", "Advanced"],
                    index=["Beginner", "Intermediate", "Advanced"].index(st.session_state["profile_fitness_level"])
                )

            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state["profile_weekly_goal"] = st.number_input(
                        "Weekly Workout Goal (days)",
                        min_value=1,
                        max_value=7,
                        value=st.session_state["profile_weekly_goal"]
                    )
                    st.session_state["profile_preferred_exercises"] = st.multiselect(
                        "Preferred Exercises",
                        ["Push-ups", "Pull-ups", "Squats", "Lunges", "Planks", "Burpees", "Jumping Jacks", "Mountain Climbers"],
                        default=st.session_state["profile_preferred_exercises"]
                    )
                with col2:
                    st.session_state["profile_target_weight"] = st.number_input(
                        "Target Weight (kg)",
                        min_value=1.0,
                        max_value=300.0,
                        value=st.session_state["profile_target_weight"]
                    )
                    st.session_state["profile_workout_duration"] = st.number_input(
                        "Preferred Workout Duration (minutes)",
                        min_value=5,
                        max_value=180,
                        value=st.session_state["profile_workout_duration"]
                    )
                st.session_state["profile_notes"] = st.text_area(
                    "Additional Notes/Goals",
                    value=st.session_state["profile_notes"],
                    placeholder="Enter any specific goals or notes about your fitness journey..."
                )

            with tab3:
                st.subheader("Achievements")
                achievements = st.session_state["profile_achievements"]
                remove_indices = []
                for idx, ach in enumerate(achievements):
                    with st.expander(f"Achievement {idx+1}: {ach.get('title', '')}"):
                        col1, col2 = st.columns([2,1])
                        with col1:
                            title = st.text_input(f"Title {idx}", value=ach.get("title", ""), key=f"ach_title_{idx}")
                            description = st.text_area(f"Description {idx}", value=ach.get("description", ""), key=f"ach_desc_{idx}")
                            date = st.date_input(f"Date {idx}", value=pd.to_datetime(ach.get("date", datetime.now().date())), key=f"ach_date_{idx}")
                        with col2:
                            if st.form_submit_button(f"Remove Achievement {idx+1}"):
                                remove_indices.append(idx)
                        # Update achievement in session state
                        achievements[idx] = {
                            "title": title,
                            "description": description,
                            "date": str(date)
                        }
                # Remove marked achievements
                for idx in sorted(remove_indices, reverse=True):
                    del achievements[idx]
                if st.form_submit_button("Add Achievement"):
                    achievements.append({"title": "", "description": "", "date": str(datetime.now().date())})
                st.session_state["profile_achievements"] = achievements

            submit = st.form_submit_button("Update Profile")
            if submit:
                profile_data = collect_profile_data()
                if update_profile(profile_data):
                    st.success("Profile updated successfully!")
                else:
                    st.error("Failed to update profile. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the server. Please try again later.")
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def calculate_streak(workouts):
    """Calculate the current workout streak from workout history."""
    if not workouts:
        return 0
        
    # Filter workouts with valid date field and sort by date in descending order
    valid_workouts = [w for w in workouts if 'date' in w and w['date']]
    if not valid_workouts:
        return 0
        
    try:
        # Sort workouts by date in descending order
        sorted_workouts = sorted(valid_workouts, key=lambda x: x['date'], reverse=True)
        
        # Convert dates to datetime objects
        workout_dates = [datetime.strptime(w['date'], '%Y-%m-%d').date() for w in sorted_workouts]
        
        # Get today's date
        today = datetime.now().date()
        
        # Initialize streak counter
        streak = 0
        last_date = today
        
        # Calculate streak
        for workout_date in workout_dates:
            # If more than one day gap, break the streak
            if (last_date - workout_date).days > 1:
                break
                
            # If it's a new day (not the same day as previous workout)
            if workout_date != last_date:
                streak += 1
                last_date = workout_date
                
        return streak
    except (ValueError, TypeError) as e:
        # Handle invalid date formats
        st.error(f"Error calculating streak: {str(e)}")
        return 0

def calculate_goal_progress(profile: dict) -> float:
    """Calculate progress towards fitness goal"""
    if profile['fitness_goal'] == "Weight Loss":
        if profile.get('target_weight') and profile.get('weight'):
            start_weight = profile.get('start_weight', profile['weight'])
            current_weight = profile['weight']
            target_weight = profile['target_weight']
            
            if start_weight > target_weight:
                return min(1.0, max(0.0, (start_weight - current_weight) / (start_weight - target_weight)))
    
    elif profile['fitness_goal'] == "Muscle Gain":
        if profile.get('target_weight') and profile.get('weight'):
            start_weight = profile.get('start_weight', profile['weight'])
            current_weight = profile['weight']
            target_weight = profile['target_weight']
            
            if target_weight > start_weight:
                return min(1.0, max(0.0, (current_weight - start_weight) / (target_weight - start_weight)))
    
    # Default progress (based on workouts completed)
    return 0.5  # Placeholder value

# Initialize session state variables if not exists
if 'page_init' not in st.session_state:
    st.session_state.page_init = False
    st.session_state.exercise_running = False
    st.session_state.exercise_completed = False
    st.session_state.exercise_data = None

def toggle_exercise_state():
    st.session_state.exercise_running = not st.session_state.exercise_running
    if not st.session_state.exercise_running:
        st.session_state.exercise_completed = True

def show_exercise_page():
    st.title("💪 Exercise Tracker")
    
    # Initialize page state if not done
    if not st.session_state.page_init:
        st.session_state.exercise_running = False
        st.session_state.exercise_completed = False
        st.session_state.exercise_data = None
        st.session_state.page_init = True
    
    # Exercise selection
    exercise_type = st.selectbox(
        "Select Exercise",
        EXERCISE_TYPES,
        help="Choose the exercise you want to perform"
    )
    
    # Exercise instructions
    exercise_instructions = {
    "Push-ups": [
        "Start in a plank position with arms shoulder-width apart",
        "Keep your body in a straight line from head to heels",
        "Lower your body until your chest nearly touches the ground",
        "Push back up to the starting position",
        "Keep your core tight throughout the movement"
    ],
    "Squats": [
        "Stand with feet shoulder-width apart",
        "Keep your chest up and back straight",
        "Lower your body as if sitting back into a chair",
        "Keep your knees in line with your toes",
        "Go down until your thighs are parallel to the ground"
    ],
    "Lunges": [
        "Stand with feet hip-width apart",
        "Step forward with one leg",
        "Lower your body until both knees are bent at 90 degrees",
        "Keep your front knee aligned with your ankle",
        "Keep your torso upright throughout the movement"
    ],
    "Plank": [
        "Start on your elbows and toes with your body in a straight line",
        "Engage your core and glutes",
        "Keep your neck and spine neutral",
        "Hold the position without letting your hips drop",
        "Maintain even breathing throughout"
    ],
    "Bicep Curls": [
        "Stand with feet shoulder-width apart, arms at your sides",
        "Hold weights with palms facing forward",
        "Bend your elbows to lift the weights toward your shoulders",
        "Lower the weights slowly back to the starting position",
        "Keep your upper arms stationary during the movement"
    ],
    "Shoulder Press": [
        "Hold weights at shoulder height with palms facing forward",
        "Press the weights upward until your arms are fully extended",
        "Keep your back straight and avoid arching",
        "Lower the weights slowly back to shoulder level",
        "Engage your core for balance and stability"
    ],
    "Deadlifts": [
        "Stand with feet hip-width apart, barbell over your midfoot",
        "Bend at your hips and knees to grip the barbell",
        "Keep your back flat and chest up",
        "Lift the bar by straightening your hips and knees",
        "Lower the bar to the ground by pushing your hips back"
    ],
    "Jumping Jacks": [
        "Stand upright with your legs together and arms at your sides",
        "Jump while spreading your legs and raising your arms overhead",
        "Land softly with your knees slightly bent",
        "Jump back to the starting position",
        "Repeat the movement at a steady pace"
    ]
 }

    # Show instructions for selected exercise
    with st.expander("📝 Exercise Instructions", expanded=True):
        if exercise_type in exercise_instructions:
            for instruction in exercise_instructions[exercise_type]:
                st.write(f"• {instruction}")
        else:
            st.write("• Follow proper form and technique for this exercise")
    
    # Exercise tracking mode selection
    tracking_mode = st.radio(
        "Choose tracking mode",
        ["Live Video Analysis", "Upload Video"],
        horizontal=True
    )
    
    if tracking_mode == "Live Video Analysis":
        # Start/Stop button
        col1, col2 = st.columns([1, 3])
        with col1:
            button_text = "⏹️ Stop" if st.session_state.exercise_running else "▶️ Start"
            button_type = "secondary" if st.session_state.exercise_running else "primary"
            
            if st.button(
                button_text,
                key="exercise_control_button",
                use_container_width=True,
                type=button_type,
                on_click=toggle_exercise_state
            ):
                st.rerun()
        
        # Live exercise tracking
        if st.session_state.exercise_running:
            try:
                # Create placeholders for video and stats
                video_col, stats_col = st.columns([3, 1])
                
                with video_col:
                    video_placeholder = st.empty()
                    
                with stats_col:
                    st.markdown("### Live Stats")
                    rep_counter = st.empty()
                    form_score = st.empty()
                    calories = st.empty()
                    feedback = st.empty()
                
                # Initialize webcam
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Failed to access webcam. Please check your camera settings.")
                    st.session_state.exercise_running = False
                    st.rerun()
                    return
                
                # Initialize exercise detector
                detector = ExerciseDetector()
                detector.reset_counters()
                
                start_time = time.time()
                
                while st.session_state.exercise_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam")
                        break
                    
                    # Detect pose and analyze form
                    pose_data = detector.detect_pose(frame)
                    if pose_data:
                        analysis = detector.analyze_exercise(exercise_type, pose_data['landmarks'])
                        
                        # Draw feedback on frame
                        frame = detector.draw_feedback(frame, analysis)
                        
                        # Update stats
                        duration = int(time.time() - start_time)
                        calories_burned = duration * EXERCISE_CALORIES.get(exercise_type, 5) / 60
                        
                        # Update UI
                        rep_counter.metric(
                            "Repetitions",
                            f"✅ {analysis['reps']} | ❌ {analysis['incorrect_reps']}"
                        )
                        form_score.metric("Form Score", f"{analysis['score']}%")
                        calories.metric("Calories", f"{calories_burned:.1f}")
                        
                        if analysis['feedback']:
                            feedback.warning('\n'.join(analysis['feedback']))
                        else:
                            feedback.success("Good form!")
                        
                        # Display frame
                        video_placeholder.image(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            channels="RGB",
                            use_container_width=True
                        )
                        
                        # Store exercise data for summary
                        st.session_state.exercise_data = {
                            "username": st.session_state.username,
                            "exercise_type": exercise_type,
                            "duration": duration,
                            "calories_burned": calories_burned,
                            "form_score": analysis['score'],
                            "feedback": analysis['feedback'],
                            "reps": analysis['reps'],
                            "incorrect_reps": analysis['incorrect_reps'],
                            "timestamp": datetime.utcnow().isoformat()
                        }
                
                cap.release()
                
            except Exception as e:
                st.error(f"Error during exercise tracking: {str(e)}")
                st.session_state.exercise_running = False
                if 'cap' in locals():
                    cap.release()
    
    else:  # Upload Video mode
        uploaded_file = st.file_uploader(
            "Upload your exercise video",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video of your exercise to analyze form"
        )
        
        if uploaded_file:
            # Save video temporarily
            temp_file = f"temp_video_{uploaded_file.name}"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            if st.button("Analyze Video", type="primary"):
                with st.spinner("Analyzing video..."):
                    try:
                        # Initialize exercise detector
                        detector = ExerciseDetector()
                        detector.reset_counters()
                        
                        # Process video
                        cap = cv2.VideoCapture(temp_file)
                        if not cap.isOpened():
                            st.error("Failed to open video file")
                            return
                            
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        duration = total_frames / fps
                        
                        progress_bar = st.progress(0)
                        frame_placeholder = st.empty()
                        
                        total_score = 0
                        total_frames_analyzed = 0
                        feedback_list = []
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            progress = current_frame / total_frames
                            progress_bar.progress(progress)
                            
                            pose_data = detector.detect_pose(frame)
                            if pose_data:
                                analysis = detector.analyze_exercise(exercise_type, pose_data['landmarks'])
                                frame = detector.draw_feedback(frame, analysis)
                                
                                total_score += analysis['score']
                                total_frames_analyzed += 1
                                if analysis['feedback']:
                                    feedback_list.extend(analysis['feedback'])
                                
                                frame_placeholder.image(
                                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    channels="RGB",
                                    use_container_width=True
                                )
                        
                        cap.release()
                        os.remove(temp_file)  # Clean up temporary file
                        
                        # Calculate final metrics
                        avg_score = total_score / total_frames_analyzed if total_frames_analyzed > 0 else 0
                        calories_burned = duration * EXERCISE_CALORIES.get(exercise_type, 5) / 60
                        unique_feedback = list(set(feedback_list))  # Remove duplicate feedback
                        
                        # Save exercise data
                        st.session_state.exercise_data = {
                            "username": st.session_state.username,
                            "exercise_type": exercise_type,
                            "duration": int(duration),
                            "calories_burned": calories_burned,
                            "form_score": avg_score,
                            "feedback": unique_feedback[:5],  # Top 5 most important feedback points
                            "reps": detector.get_rep_count(),
                            "incorrect_reps": detector.get_incorrect_rep_count(),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        st.session_state.exercise_completed = True
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error analyzing video: {str(e)}")
                        if 'cap' in locals():
                            cap.release()
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
    
    # Show exercise summary after completion
    if st.session_state.exercise_completed and st.session_state.exercise_data:
        st.markdown("---")
        st.markdown("### Exercise Summary")

        # Only upload if not already uploaded
        if not st.session_state.get('exercise_uploaded', False):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/exercise/",
                    json=st.session_state.exercise_data,
                    headers=get_auth_headers(),
                    timeout=10
                )
                st.session_state.exercise_uploaded = (response.status_code == 200)
                if response.status_code != 200:
                    st.error(f"Failed to save exercise data: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error saving exercise data: {str(e)}")
                st.button("Retry Saving Data", type="primary", on_click=lambda: None)

        if st.session_state.get('exercise_uploaded', False):
            # Display summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("correct Reps", st.session_state.exercise_data['reps'])
            with col2:
                st.metric("Incorrect Reps", st.session_state.exercise_data['incorrect_reps'])
            with col3:
                st.metric("Form Score", f"{st.session_state.exercise_data['form_score']:.1f}%")
            with col4:
                st.metric("Duration", f"{st.session_state.exercise_data['duration']} sec")
            with col5:
                st.metric("Calories", f"{st.session_state.exercise_data['calories_burned']:.1f}")

            # Form feedback
            if st.session_state.exercise_data['feedback']:
                st.markdown("#### Form Feedback")
                for feedback in st.session_state.exercise_data['feedback']:
                    st.warning(feedback)
            else:
                st.success("Great form! Keep up the good work!")

            # Reset session state
            if st.button("Start New Exercise", type="primary"):
                st.session_state.exercise_completed = False
                st.session_state.exercise_data = None
                st.session_state.exercise_uploaded = False
                st.rerun()

def show_workout_plans():
    st.title("📋 Workout Plans")
    
    # Create new workout plan
    with st.form("workout_plan_form"):
        st.subheader("Create New Workout Plan")
        
        exercises = []
        num_exercises = st.number_input("Number of Exercises", min_value=1, max_value=10, value=3)
        
        for i in range(num_exercises):
            col1, col2 = st.columns(2)
            with col1:
                exercise_type = st.selectbox(f"Exercise {i+1}", EXERCISE_TYPES, key=f"exercise_{i}")
            with col2:
                duration = st.number_input(f"Duration (min)", min_value=1, max_value=30, value=10, key=f"duration_{i}")
            exercises.append({"type": exercise_type, "duration": duration})
        
        total_duration = sum(ex["duration"] for ex in exercises)
        difficulty = st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced"])
        
        submit_button = st.form_submit_button("Create Workout Plan")
        
        if submit_button:
            try:
                plan_data = {
                    "username": st.session_state.username,
                    "exercises": exercises,
                    "duration": total_duration,
                    "difficulty": difficulty,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/workout/plan",
                    json=plan_data,
                    headers=get_auth_headers()
                )
                
                if response.status_code == 200:
                    st.success("Workout plan created successfully!")
                else:
                    st.error("Failed to create workout plan")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the server. Please try again later.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display existing workout plans
    st.subheader("Your Workout Plans")
    try:
        response = requests.get(
            f"{API_BASE_URL}/workout/plan/{st.session_state.username}",
            headers=get_auth_headers()
        )
        
        if response.status_code == 200:
            plans = response.json()
            
            if plans:
                for plan in plans:
                    with st.expander(f"Plan ({plan['difficulty']} - {plan['duration']} min)"):
                        st.write("Exercises:")
                        for exercise in plan["exercises"]:
                            st.write(f"- {exercise['type']}: {exercise['duration']} min")
                        st.write(f"Created: {plan['created_at']}")
            else:
                st.info("No workout plans available yet")
        else:
            st.error("Failed to load workout plans")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the server. Please try again later.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def show_profile():
    st.title("My Profile")
    
    # Get user profile data
    profile_data = get_profile_data()
    if not profile_data:
        st.error("Could not load profile data")
        return
        
    # Progress metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        streak = calculate_streak(profile_data.get('workout_dates', []))
        st.metric("Current Streak", f"{streak} days")
        
    with col2:
        weekly_goal = profile_data.get('weekly_workout_goal', 0)
        workouts_this_week = len([d for d in profile_data.get('workout_dates', []) 
                                if (datetime.now() - datetime.strptime(d, "%Y-%m-%d")).days <= 7])
        progress = min(100, int((workouts_this_week / weekly_goal * 100) if weekly_goal else 0))
        st.metric("Weekly Progress", f"{workouts_this_week}/{weekly_goal}")
        st.progress(progress/100)
        
    with col3:
        current_weight = profile_data.get('current_weight', 0)
        target_weight = profile_data.get('target_weight', 0)
        if current_weight and target_weight:
            weight_diff = target_weight - current_weight
            st.metric("Weight Progress", 
                     f"{current_weight}kg", 
                     f"{weight_diff:+.1f}kg to goal")
    
    # Weight history chart
    weight_history = profile_data.get('weight_history', [])
    if weight_history:
        st.subheader("Weight History")
        df = pd.DataFrame(weight_history)
        st.line_chart(df.set_index('date')['weight'])
    
    # Recent workouts
    st.subheader("Recent Workouts")
    recent_workouts = profile_data.get('recent_workouts', [])
    if recent_workouts:
        for workout in recent_workouts[:5]:
            with st.expander(f"{workout['date']} - {workout['type']}"):
                st.write(f"Duration: {workout['duration']} minutes")
                st.write(f"Exercises: {', '.join(workout['exercises'])}")
    else:
        st.info("No recent workouts found")
        
    # Profile form
    st.subheader("Update Profile")
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            weekly_goal = st.number_input("Weekly Workout Goal", 
                                        min_value=1, max_value=7,
                                        value=profile_data.get('weekly_workout_goal', 3))
            current_weight = st.number_input("Current Weight (kg)",
                                           min_value=30.0, max_value=200.0,
                                           value=profile_data.get('current_weight', 70.0))
            target_weight = st.number_input("Target Weight (kg)",
                                          min_value=30.0, max_value=200.0,
                                          value=profile_data.get('target_weight', 70.0))
                                          
        with col2:
            preferred_duration = st.number_input("Preferred Workout Duration (minutes)",
                                               min_value=15, max_value=180,
                                               value=profile_data.get('preferred_duration', 45))
            preferred_exercises = st.multiselect("Preferred Exercises",
                                               ["Running", "Cycling", "Swimming", "Weight Training",
                                                "Yoga", "HIIT", "Pilates", "Boxing"],
                                               default=profile_data.get('preferred_exercises', []))
                                               
        notes = st.text_area("Additional Notes", value=profile_data.get('notes', ''))
        
        if st.form_submit_button("Update Profile"):
            updated_data = {
                'weekly_workout_goal': weekly_goal,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'preferred_duration': preferred_duration,
                'preferred_exercises': preferred_exercises,
                'notes': notes
            }
            
            success = update_profile(updated_data)
            if success:
                st.success("Profile updated successfully!")
            else:
                st.error("Failed to update profile. Please try again.")

def get_workout_history():
    """Retrieve workout history from the backend."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/workouts/history/{st.session_state.username}",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # No history found for user, return empty list
            return []
        else:
            st.error(f"Failed to retrieve workout history: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error retrieving workout history: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="AI Fitness Tracker",
        page_icon="💪",
        layout="wide"
    )
    
    # Initialize session state variables
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Show login page if not authenticated
    if not st.session_state.access_token:
        login()
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        st.write(f"👤 Welcome : {st.session_state.username}")
        
        # Navigation buttons
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_page = "Home"
            st.rerun()
        if st.button("💪 Exercise Tracker", use_container_width=True):
            st.session_state.current_page = "Exercise Tracker"
            st.rerun()
        if st.button("👤 Profile", use_container_width=True):
            st.session_state.current_page = "Profile"
            st.rerun()
        if st.button("📋 Workout Plans", use_container_width=True):
            st.session_state.current_page = "Workout Plans"
            st.rerun()
        
        # Add logout button at the bottom of sidebar
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.access_token = None
            st.session_state.username = None
            st.session_state.current_page = "Home"
            st.rerun()
    
    # Show selected page
    if st.session_state.current_page == "Home":
        show_home_page()
    elif st.session_state.current_page == "Exercise Tracker":
        show_exercise_page()
    elif st.session_state.current_page == "Profile":
        show_profile_page()
    elif st.session_state.current_page == "Workout Plans":
        show_workout_plans()

if __name__ == "__main__":
    main() 