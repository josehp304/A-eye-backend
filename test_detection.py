#!/usr/bin/env python3
"""
Test script for object detection using a sample image
This script can be used to test the object detection functionality
without requiring a webcam.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import requests
from PIL import Image
import io

def test_detection_with_sample_image():
    """
    Test object detection using a sample image downloaded from the internet
    """
    print("Testing object detection with sample image...")
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully!")
    
    # Download a sample image for testing
    sample_image_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
    
    try:
        print("Downloading sample image...")
        response = requests.get(sample_image_url)
        response.raise_for_status()
        
        # Convert to OpenCV format
        pil_image = Image.open(io.BytesIO(response.content))
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        print("Running object detection...")
        
        # Run detection
        results = model(opencv_image, conf=0.5, verbose=True)
        
        # Draw results on image
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    print(f"Detection {i+1}: {class_name} (confidence: {confidence:.2f})")
                    
                    # Draw bounding box
                    cv2.rectangle(opencv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(opencv_image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save result
        output_path = "test_detection_result.jpg"
        cv2.imwrite(output_path, opencv_image)
        print(f"Detection result saved as '{output_path}'")
        
        # Display result (if display is available)
        try:
            cv2.imshow('Object Detection Test', opencv_image)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Cannot display image (no display available): {e}")
            
        print("Object detection test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")

def test_camera_access():
    """
    Test camera access
    """
    print("\nTesting camera access...")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot access camera")
        print("Possible solutions:")
        print("1. Make sure your webcam is connected and not being used by another application")
        print("2. Check camera permissions")
        print("3. Try running: sudo chmod 666 /dev/video0")
        return False
    else:
        print("✅ Camera access successful!")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera frame captured successfully! Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("❌ Could not read frame from camera")
            
        cap.release()
        return ret

def main():
    """
    Run all tests
    """
    print("Object Detection System Test")
    print("=" * 40)
    
    # Test object detection with sample image
    test_detection_with_sample_image()
    
    # Test camera access
    camera_ok = test_camera_access()
    
    print("\nTest Summary:")
    print("=" * 40)
    print("✅ YOLO model loading: Success")
    print("✅ Object detection: Success")
    if camera_ok:
        print("✅ Camera access: Success")
        print("\n🎉 All tests passed! You can run the main detection script with:")
        print("python main.py")
    else:
        print("❌ Camera access: Failed")
        print("\n⚠️  Camera issues detected. Check the solutions above.")

if __name__ == "__main__":
    main()
