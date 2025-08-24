#!/usr/bin/env python3
"""
Web application to display processed video from OpenCV
Creates a simple web interface to view the object detection video stream
"""

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import time
import os
import requests
import base64
import json
from dotenv import load_dotenv
import threading
from io import BytesIO
from PIL import Image

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type", "Authorization"], methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

class WebObjectDetector:
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the web-based object detector
        """
        print("Loading YOLO model...")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
        # Colors for different classes (BGR format for OpenCV)
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (255, 192, 203), (0, 128, 0),
            (128, 128, 0), (0, 0, 128), (128, 0, 0), (0, 128, 128), (192, 192, 192)
        ]
        
        # Groq API configuration
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Person detection tracking
        self.last_person_analysis_time = 0
        self.analysis_cooldown = 10  # Seconds between analyses
        self.last_analysis = "No analysis yet"
        self.analysis_in_progress = False
        
        # Camera setup (now supports both webcam and stream input)
        self.cap = None
        self.fps_counter = 0
        self.fps_display = 0
        self.last_fps_time = time.time()
        
        # Stream input handling
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.use_webcam = False  # Default to webcam, can be changed to stream input
        self.stream_active = False
        
        print("Model loaded successfully!")
        
        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            print("⚠️  Warning: Groq API key not found. AI analysis will be disabled.")
        else:
            print("✅ Groq API key loaded successfully!")
    
    def set_frame_from_stream(self, frame_data):
        """Set current frame from incoming stream data"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                with self.frame_lock:
                    self.current_frame = frame
                    self.stream_active = True
                return True
            return False
        except Exception as e:
            print(f"Error processing incoming frame: {e}")
            return False
    
    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 encoded string"""
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    
    def analyze_frame_with_groq(self, frame):
        """Send frame to Groq API for analysis when a person is detected"""
        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            return "Groq API key not configured"
        
        try:
            image_data_url = self.frame_to_base64(frame)
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this security camera frame. Describe what you see, focusing on any people detected, their activities, and any potential security concerns. Be concise but detailed."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                }
                            }
                        ]
                    }
                ],
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "temperature": 0.7,
                "max_completion_tokens": 300,
                "top_p": 1,
                "stream": False,
                "stop": None
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.groq_api_key}"
            }
            
            response = requests.post(self.groq_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content']
                return analysis
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error analyzing frame: {str(e)}"
    
    def detect_objects(self, frame):
        """Detect objects in a frame"""
        person_detected = False
        
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Get confidence and class
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Check if person is detected
                    if class_name == "person":
                        person_detected = True
                    
                    # Choose color for this class
                    color = self.colors[class_id % len(self.colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label with class name and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Calculate label size and position
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_height - baseline - 5),
                        (x1 + label_width, y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
        
        return frame, person_detected
    
    def start_camera(self):
        """Initialize camera"""
        if not self.use_webcam:
            return True  # Stream mode doesn't need camera initialization
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def stop_camera(self):
        """Release camera"""
        if self.cap:
            self.cap.release()
    
    def generate_frames(self):
        """Generate video frames for streaming"""
        if not self.start_camera():
            return
        
        try:
            while True:
                if self.use_webcam:
                    # Read from webcam
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                else:
                    # Read from stream input
                    with self.frame_lock:
                        if self.current_frame is None:
                            time.sleep(0.033)  # Wait ~30ms if no frame available
                            continue
                        frame = self.current_frame.copy()
                
                # Detect objects in the frame
                frame_with_detections, person_detected = self.detect_objects(frame)
                
                # Check if we should analyze the frame with Groq
                current_time = time.time()
                if (person_detected and 
                    current_time - self.last_person_analysis_time > self.analysis_cooldown and
                    not self.analysis_in_progress and
                    self.groq_api_key and self.groq_api_key != "your_groq_api_key_here"):
                    
                    print("🤖 Person detected! Analyzing frame with Groq AI...")
                    self.analysis_in_progress = True
                    self.last_person_analysis_time = current_time
                    
                    # Run analysis in a separate thread
                    def analyze_async():
                        self.last_analysis = self.analyze_frame_with_groq(frame_with_detections)
                        self.analysis_in_progress = False
                        print("🔍 AI Analysis completed")
                    
                    threading.Thread(target=analyze_async, daemon=True).start()
                
                # Calculate FPS
                self.fps_counter += 1
                if current_time - self.last_fps_time >= 1.0:
                    self.fps_display = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                
                # Add overlay information
                cv2.putText(frame_with_detections, f"FPS: {self.fps_display}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.putText(frame_with_detections, f"Confidence: {self.confidence_threshold:.2f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add source indicator
                source_text = "Source: Webcam" if self.use_webcam else "Source: Stream"
                cv2.putText(frame_with_detections, source_text, (10, frame_with_detections.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add AI analysis status
                status_color = (0, 255, 255) if self.analysis_in_progress else (255, 255, 255)
                status_text = "AI Analyzing..." if self.analysis_in_progress else "AI Ready"
                if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
                    status_text = "AI Disabled"
                    status_color = (128, 128, 128)
                
                cv2.putText(frame_with_detections, status_text, (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame_with_detections)
                frame = buffer.tobytes()
                
                # Yield frame in byte format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
        except Exception as e:
            print(f"Error in generate_frames: {e}")
        finally:
            self.stop_camera()

# Initialize the detector
detector = WebObjectDetector()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(detector.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_input', methods=['POST'])
def stream_input():
    """Endpoint to receive MJPEG stream from frontend"""
    try:
        # Get the image data from the request
        image_data = request.get_data()
        
        if detector.set_frame_from_stream(image_data):
            # Switch to stream mode if not already
            if detector.use_webcam:
                detector.use_webcam = False
                print("🔄 Switched to stream input mode")
            
            return jsonify({'success': True, 'message': 'Frame received'})
        else:
            return jsonify({'success': False, 'error': 'Failed to process frame'}), 400
            
    except Exception as e:
        print(f"Error in stream_input: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/switch_source/<source>')
def switch_source(source):
    """Switch between webcam and stream input"""
    if source == 'webcam':
        detector.use_webcam = True
        return jsonify({'success': True, 'source': 'webcam'})
    elif source == 'stream':
        detector.use_webcam = False
        return jsonify({'success': True, 'source': 'stream'})
    else:
        return jsonify({'success': False, 'error': 'Invalid source. Use "webcam" or "stream"'})

@app.route('/api/status')
def status():
    """API endpoint to get current status"""
    return jsonify({
        'fps': detector.fps_display,
        'confidence': detector.confidence_threshold,
        'ai_analysis': detector.last_analysis,
        'ai_in_progress': detector.analysis_in_progress,
        'ai_enabled': bool(detector.groq_api_key and detector.groq_api_key != "your_groq_api_key_here"),
        'source': 'webcam' if detector.use_webcam else 'stream',
        'stream_active': detector.stream_active
    })

@app.route('/api/set_confidence/<float:confidence>')
def set_confidence(confidence):
    """API endpoint to set confidence threshold"""
    if 0.1 <= confidence <= 1.0:
        detector.confidence_threshold = confidence
        return jsonify({'success': True, 'confidence': confidence})
    return jsonify({'success': False, 'error': 'Confidence must be between 0.1 and 1.0'})

if __name__ == '__main__':
    print("Starting Web Object Detection Server...")
    print("Visit http://localhost:5000 to view the video stream")
    print("The server can accept both webcam input and MJPEG streams from frontend")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
