import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple

class ExerciseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.rep_counter = 0
        self.incorrect_reps = 0
        self.stage = None
        self.form_feedback = []
    
    def detect_pose(self, frame) -> Dict:
        """Detect human pose in the frame and return landmark info"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = [
            {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            }
            for lm in results.pose_landmarks.landmark
        ]
        
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        return {
            'landmarks': landmarks,
            'frame_height': frame.shape[0],
            'frame_width': frame.shape[1]
        }
    
    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def analyze_pushup(self, landmarks: List[Dict]) -> Dict:
        """Analyze push-up exercise"""
        left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
        left_elbow = (landmarks[13]['x'], landmarks[13]['y'])
        left_wrist = (landmarks[15]['x'], landmarks[15]['y'])
        right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
        right_elbow = (landmarks[14]['x'], landmarks[14]['y'])
        right_wrist = (landmarks[16]['x'], landmarks[16]['y'])
        left_hip = (landmarks[23]['x'], landmarks[23]['y'])
        
        left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        body_angle = self.calculate_angle(left_shoulder, left_hip, (left_hip[0], left_hip[1] + 0.5))
        
        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
        score = 100
        feedback = []
        is_correct_form = True

        if not (70 <= left_arm_angle <= 110 and 70 <= right_arm_angle <= 110):
            score -= 30
            feedback.append("Maintain arms at ~90° angle during push-up")
            is_correct_form = False

        if body_angle < 160:
            score -= 30
            feedback.append("Keep body straight (no sagging or arching)")
            is_correct_form = False

        # Rep Counting
        if avg_arm_angle < 90 and self.stage != 'down':
            self.stage = 'down'
        elif avg_arm_angle > 160 and self.stage == 'down':
            self.stage = 'up'
            if is_correct_form:
                self.rep_counter += 1
            else:
                self.incorrect_reps += 1

        return {
            'score': max(0, score),
            'feedback': feedback,
            'angles': {
                'left_arm_angle': left_arm_angle,
                'right_arm_angle': right_arm_angle,
                'body_angle': body_angle
            },
            'reps': self.rep_counter,
            'incorrect_reps': self.incorrect_reps,
            'stage': self.stage
        }
    
    def analyze_squat(self, landmarks: List[Dict]) -> Dict:
        """Analyze squat exercise"""
        left_hip = (landmarks[23]['x'], landmarks[23]['y'])
        left_knee = (landmarks[25]['x'], landmarks[25]['y'])
        left_ankle = (landmarks[27]['x'], landmarks[27]['y'])
        
        right_hip = (landmarks[24]['x'], landmarks[24]['y'])
        right_knee = (landmarks[26]['x'], landmarks[26]['y'])
        right_ankle = (landmarks[28]['x'], landmarks[28]['y'])
        
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        hip_angle = self.calculate_angle((landmarks[11]['x'], landmarks[11]['y']), left_hip, left_knee)
        
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        score = 100
        feedback = []
        is_correct_form = True
        
        if avg_knee_angle < 70:
            score -= 30
            feedback.append("Don't squat too low")
            is_correct_form = False
        elif avg_knee_angle > 150:
            score -= 30
            feedback.append("Go lower into your squat")
            is_correct_form = False

        if hip_angle < 45:
            score -= 20
            feedback.append("Keep your back straight")
            is_correct_form = False

        # Rep Counting
        if avg_knee_angle < 110 and self.stage != 'down':
            self.stage = 'down'
        elif avg_knee_angle > 160 and self.stage == 'down':
            self.stage = 'up'
            if is_correct_form:
                self.rep_counter += 1
            else:
                self.incorrect_reps += 1

        return {
            'score': max(0, score),
            'feedback': feedback,
            'angles': {
                'left_knee_angle': left_knee_angle,
                'right_knee_angle': right_knee_angle,
                'hip_angle': hip_angle
            },
            'reps': self.rep_counter,
            'incorrect_reps': self.incorrect_reps,
            'stage': self.stage
        }
    
    def analyze_lunge(self, landmarks: List[Dict]) -> Dict:
        """Analyze lunge exercise"""
        left_hip = (landmarks[23]['x'], landmarks[23]['y'])
        left_knee = (landmarks[25]['x'], landmarks[25]['y'])
        left_ankle = (landmarks[27]['x'], landmarks[27]['y'])
        right_hip = (landmarks[24]['x'], landmarks[24]['y'])
        right_knee = (landmarks[26]['x'], landmarks[26]['y'])
        right_ankle = (landmarks[28]['x'], landmarks[28]['y'])
        front_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        back_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        torso_angle = self.calculate_angle((landmarks[11]['x'], landmarks[11]['y']), left_hip, (left_hip[0], left_hip[1] + 0.5))
        avg_knee_angle = (front_knee_angle + back_knee_angle) / 2
        score = 100
        feedback = []
        is_correct_form = True

        if not (85 <= front_knee_angle <= 95):
            score -= 30
            feedback.append("Front knee must be ~90°")
            is_correct_form = False

        if not (85 <= back_knee_angle <= 95):
            score -= 30
            feedback.append("Back knee must be ~90°")
            is_correct_form = False

        if torso_angle < 80:
            score -= 20
            feedback.append("Keep torso upright")
            is_correct_form = False

        # Rep Counting
        if avg_knee_angle < 100 and self.stage != 'down':
            self.stage = 'down'
        elif avg_knee_angle > 160 and self.stage == 'down':
            self.stage = 'up'
            if is_correct_form:
                self.rep_counter += 1
            else:
                self.incorrect_reps += 1

        return {
            'score': max(0, score),
            'feedback': feedback,
            'angles': {
                'front_knee_angle': front_knee_angle,
                'back_knee_angle': back_knee_angle,
                'torso_angle': torso_angle
            },
            'reps': self.rep_counter,
            'incorrect_reps': self.incorrect_reps,
            'stage': self.stage
        }
    
    def analyze_plank(self, landmarks: List[dict]) -> dict:
        """Analyze plank exercise (template for new exercises)."""
        left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
        left_hip = (landmarks[23]['x'], landmarks[23]['y'])
        left_ankle = (landmarks[27]['x'], landmarks[27]['y'])
        body_angle = self.calculate_angle(left_shoulder, left_hip, left_ankle)
        score = 100
        feedback = []
        is_correct_form = True

        if body_angle < 160:
            score -= 30
            feedback.append("Keep your body straight in plank!")
            is_correct_form = False

        # No rep counting for plank, but track form
        return {
            'score': max(0, score),
            'feedback': feedback,
            'angles': {'body_angle': body_angle},
            'reps': self.rep_counter,
            'incorrect_reps': self.incorrect_reps,
            'stage': self.stage
        }

    def analyze_default(self, landmarks: List[dict]) -> dict:
        """Default analysis for unsupported exercises."""
        return {
            'score': 0,
            'feedback': ["Exercise not supported yet."],
            'angles': {},
            'reps': self.rep_counter,
            'incorrect_reps': self.incorrect_reps,
            'stage': self.stage
        }

    def analyze_bicep_curl(self, landmarks: List[Dict]) -> Dict:
        """Analyze bicep curl exercise (assume right arm for simplicity)"""
        right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
        right_elbow = (landmarks[14]['x'], landmarks[14]['y'])
        right_wrist = (landmarks[16]['x'], landmarks[16]['y'])
        elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        score = 100
        feedback = []
        is_correct_form = True

        if elbow_angle > 60:
            feedback.append("Curl your arm more (lower elbow angle)")
            score -= 30
            is_correct_form = False
        if elbow_angle < 30:
            feedback.append("Don't over-curl (raise elbow angle)")
            score -= 10
            is_correct_form = False

        # Rep counting
        if elbow_angle < 40 and self.stage != 'up':
            self.stage = 'up'
        elif elbow_angle > 120 and self.stage == 'up':
            self.stage = 'down'
            if is_correct_form:
                self.rep_counter += 1
            else:
                self.incorrect_reps += 1

        return {
            'score': max(0, score),
            'feedback': feedback,
            'angles': {'elbow_angle': elbow_angle},
            'reps': self.rep_counter,
            'incorrect_reps': self.incorrect_reps,
            'stage': self.stage
        }

    def analyze_shoulder_press(self, landmarks: List[Dict]) -> Dict:
        """Analyze shoulder press (assume right arm for simplicity)"""
        right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
        right_elbow = (landmarks[14]['x'], landmarks[14]['y'])
        right_wrist = (landmarks[16]['x'], landmarks[16]['y'])
        elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        score = 100
        feedback = []
        is_correct_form = True

        if elbow_angle < 160:
            feedback.append("Fully extend your arm at the top")
            score -= 30
            is_correct_form = False
        if elbow_angle > 100:
            feedback.append("Lower the weight more at the bottom")
            score -= 10
            is_correct_form = False

        # Rep counting
        if elbow_angle > 150 and self.stage != 'up':
            self.stage = 'up'
        elif elbow_angle < 90 and self.stage == 'up':
            self.stage = 'down'
            if is_correct_form:
                self.rep_counter += 1
            else:
                self.incorrect_reps += 1

        return {
            'score': max(0, score),
            'feedback': feedback,
            'angles': {'elbow_angle': elbow_angle},
            'reps': self.rep_counter,
            'incorrect_reps': self.incorrect_reps,
            'stage': self.stage
        }

    def analyze_deadlift(self, landmarks: List[Dict]) -> Dict:
        """Analyze deadlift (use hip and back angles)"""
        left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
        left_hip = (landmarks[23]['x'], landmarks[23]['y'])
        left_knee = (landmarks[25]['x'], landmarks[25]['y'])
        hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        score = 100
        feedback = []
        is_correct_form = True

        if hip_angle < 160:
            feedback.append("Stand up straight at the top")
            score -= 30
            is_correct_form = False
        if hip_angle > 200:
            feedback.append("Don't overextend your back")
            score -= 10
            is_correct_form = False

        # Rep counting
        if hip_angle < 100 and self.stage != 'down':
            self.stage = 'down'
        elif hip_angle > 160 and self.stage == 'down':
            self.stage = 'up'
            if is_correct_form:
                self.rep_counter += 1
            else:
                self.incorrect_reps += 1

        return {
            'score': max(0, score),
            'feedback': feedback,
            'angles': {'hip_angle': hip_angle},
            'reps': self.rep_counter,
            'incorrect_reps': self.incorrect_reps,
            'stage': self.stage
        }

    def analyze_jumping_jacks(self, landmarks: List[Dict]) -> Dict:
        """Analyze jumping jacks (use hand and foot distance)"""
        left_wrist = (landmarks[15]['x'], landmarks[15]['y'])
        right_wrist = (landmarks[16]['x'], landmarks[16]['y'])
        left_ankle = (landmarks[27]['x'], landmarks[27]['y'])
        right_ankle = (landmarks[28]['x'], landmarks[28]['y'])
        hand_dist = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
        foot_dist = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))
        score = 100
        feedback = []
        is_correct_form = True

        if hand_dist < 0.3:
            feedback.append("Raise your arms higher")
            score -= 30
            is_correct_form = False
        if foot_dist < 0.3:
            feedback.append("Spread your feet wider")
            score -= 30
            is_correct_form = False

        # Rep counting
        if hand_dist > 0.5 and foot_dist > 0.5 and self.stage != 'open':
            self.stage = 'open'
        elif hand_dist < 0.3 and foot_dist < 0.3 and self.stage == 'open':
            self.stage = 'close'
            if is_correct_form:
                self.rep_counter += 1
            else:
                self.incorrect_reps += 1

        return {
            'score': max(0, score),
            'feedback': feedback,
            'angles': {'hand_dist': hand_dist, 'foot_dist': foot_dist},
            'reps': self.rep_counter,
            'incorrect_reps': self.incorrect_reps,
            'stage': self.stage
        }

    def analyze_exercise(self, exercise_type: str, landmarks: List[dict]) -> dict:
        """Select correct exercise analyzer based on exercise_type."""
        if not landmarks:
            return {
                'score': 0,
                'feedback': ["No pose detected"],
                'angles': {},
                'reps': 0,
                'incorrect_reps': 0,
                'stage': None
            }
        analysis_functions = {
            'Push-ups': self.analyze_pushup,
            'Squats': self.analyze_squat,
            'Lunges': self.analyze_lunge,
            'Plank': self.analyze_plank,
            'Bicep Curls': self.analyze_bicep_curl,
            'Shoulder Press': self.analyze_shoulder_press,
            'Deadlifts': self.analyze_deadlift,
            'Jumping Jacks': self.analyze_jumping_jacks,
        }
        return analysis_functions.get(exercise_type, self.analyze_default)(landmarks)
    
    def reset_counters(self):
        """Reset counters when switching exercises."""
        self.rep_counter = 0
        self.incorrect_reps = 0
        self.stage = None
        self.form_feedback = []
    
    def draw_feedback(self, frame, analysis_result: Dict) -> np.ndarray:
        """Overlay text feedback, score, and reps"""
        y = 30
        for msg in analysis_result['feedback']:
            cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 30
        
        cv2.putText(frame, f"Correct Reps: {analysis_result['reps']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30
        
        cv2.putText(frame, f"Incorrect Reps: {analysis_result['incorrect_reps']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y += 30
        
        cv2.putText(frame, f"Form Score: {analysis_result['score']}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame

    def get_rep_count(self):
        """Return the number of correct reps detected."""
        return self.rep_counter

    def get_incorrect_rep_count(self):
        """Return the number of incorrect reps detected."""
        return self.incorrect_reps

    def draw_skeleton(self, frame, landmarks):
        """Draw skeleton overlay on the frame using pose landmarks."""
        h, w = frame.shape[:2]
        points = []
        for lm in landmarks:
            x, y = int(lm['x'] * w), int(lm['y'] * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        # Example skeleton pairs (MediaPipe indices)
        skeleton_pairs = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 12),            # Shoulders
            (23, 24),            # Hips
            (11, 23), (12, 24),  # Torso sides
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28)   # Right leg
        ]
        for i, j in skeleton_pairs:
            if i < len(points) and j < len(points):
                cv2.line(frame, points[i], points[j], (255, 0, 0), 2)
        return frame
