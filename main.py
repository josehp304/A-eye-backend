import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import requests
import base64
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the object detector with YOLO model
        
        Args:
            model_name (str): YOLO model to use (yolov8n.pt for nano, yolov8s.pt for small, etc.)
            confidence_threshold (float): Minimum confidence for detections
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
        self.analysis_cooldown = 10  # Seconds between analyses to avoid spam
        
        print("Model loaded successfully!")
        
        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            print("⚠️  Warning: Groq API key not found. AI analysis will be disabled.")
            print("   Please set your GROQ_API_KEY in the .env file to enable AI analysis.")
        else:
            print("✅ Groq API key loaded successfully!")
    
    def frame_to_base64(self, frame):
        """
        Convert OpenCV frame to base64 encoded string
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            str: Base64 encoded image data URL
        """
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create data URL
        return f"data:image/jpeg;base64,{img_base64}"
    
    def analyze_frame_with_groq(self, frame):
        """
        Send frame to Groq API for analysis when a person is detected
        
        Args:
            frame: OpenCV frame containing detected person
            
        Returns:
            str: Analysis result from Groq API
        """
        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            return "Groq API key not configured"
        
        try:
            # Convert frame to base64
            image_data_url = self.frame_to_base64(frame)
            
            # Prepare the request payload
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
            
            # Set headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.groq_api_key}"
            }
            
            # Make the API request
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
        """
        Detect objects in a frame
        
        Args:
            frame: Input image frame
            
        Returns:
            tuple: (frame with detections, person_detected flag)
        """
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
    
    def run_detection(self, camera_index=0):
        """
        Run real-time object detection using webcam
        
        Args:
            camera_index (int): Camera index (usually 0 for default webcam)
        """
        # Initialize webcam
        cap = cv2.VideoCapture(camera_index)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting real-time object detection...")
        print("Press 'q' to quit, 'c' to toggle confidence threshold, 'a' to force AI analysis")
        
        # FPS calculation variables
        prev_time = time.time()
        fps_counter = 0
        fps_display = 0
        
        # AI analysis status
        last_analysis = ""
        analysis_in_progress = False
        
        try:
            while True:
                # Read frame from camera
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Detect objects in the frame
                frame_with_detections, person_detected = self.detect_objects(frame)
                
                # Check if we should analyze the frame with Groq
                current_time = time.time()
                if (person_detected and 
                    current_time - self.last_person_analysis_time > self.analysis_cooldown and
                    not analysis_in_progress and
                    self.groq_api_key and self.groq_api_key != "your_groq_api_key_here"):
                    
                    print("🤖 Person detected! Analyzing frame with Groq AI...")
                    analysis_in_progress = True
                    self.last_person_analysis_time = current_time
                    
                    # Run analysis in a separate thread to avoid blocking
                    import threading
                    def analyze_async():
                        nonlocal last_analysis, analysis_in_progress
                        last_analysis = self.analyze_frame_with_groq(frame_with_detections)
                        analysis_in_progress = False
                        print("🔍 AI Analysis:", last_analysis)
                    
                    threading.Thread(target=analyze_async, daemon=True).start()
                
                # Calculate and display FPS
                fps_counter += 1
                
                if current_time - prev_time >= 1.0:  # Update FPS every second
                    fps_display = fps_counter
                    fps_counter = 0
                    prev_time = current_time
                
                # Add FPS text to frame
                cv2.putText(
                    frame_with_detections,
                    f"FPS: {fps_display}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Add confidence threshold text
                cv2.putText(
                    frame_with_detections,
                    f"Confidence: {self.confidence_threshold:.2f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Add AI analysis status
                status_color = (0, 255, 255) if analysis_in_progress else (255, 255, 255)
                status_text = "AI Analyzing..." if analysis_in_progress else "AI Ready"
                if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
                    status_text = "AI Disabled"
                    status_color = (128, 128, 128)
                
                cv2.putText(
                    frame_with_detections,
                    status_text,
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2
                )
                
                # Display the frame
                cv2.imshow('Real-time Object Detection', frame_with_detections)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Toggle confidence threshold between 0.3 and 0.7
                    self.confidence_threshold = 0.3 if self.confidence_threshold >= 0.5 else 0.7
                    print(f"Confidence threshold changed to: {self.confidence_threshold}")
                elif key == ord('a'):
                    # Force AI analysis
                    if not analysis_in_progress and self.groq_api_key and self.groq_api_key != "your_groq_api_key_here":
                        print("🤖 Forcing AI analysis...")
                        analysis_in_progress = True
                        
                        def analyze_async():
                            nonlocal last_analysis, analysis_in_progress
                            last_analysis = self.analyze_frame_with_groq(frame_with_detections)
                            analysis_in_progress = False
                            print("🔍 AI Analysis:", last_analysis)
                        
                        import threading
                        threading.Thread(target=analyze_async, daemon=True).start()
                    else:
                        print("❌ Cannot perform analysis: API key not configured or analysis in progress")
                elif key == ord('h'):
                    print("\nControls:")
                    print("q - Quit")
                    print("c - Toggle confidence threshold")
                    print("a - Force AI analysis")
                    print("h - Show this help")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")

def main():
    """
    Main function to run the object detection application
    """
    print("Real-time Object Detection with YOLO")
    print("=" * 40)
    
    try:
        # Initialize object detector
        detector = ObjectDetector(
            model_name='yolov8n.pt',  # Using nano model for speed
            confidence_threshold=0.5
        )
        
        # Run detection
        detector.run_detection(camera_index=0)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Make sure you have a webcam connected and the required packages installed.")
        print("Install requirements with: pip install -r requirements.txt")

if __name__ == "__main__":
    main()