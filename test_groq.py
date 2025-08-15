#!/usr/bin/env python3
"""
Test script for Groq API integration
"""

import cv2
import os
from dotenv import load_dotenv
from main import ObjectDetector

def test_groq_api():
    """
    Test the Groq API integration
    """
    print("Testing Groq API Integration")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key is configured
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        print("❌ Groq API key not found or not configured.")
        print("Please set your GROQ_API_KEY in the .env file.")
        print("\nTo get a Groq API key:")
        print("1. Visit https://console.groq.com/")
        print("2. Sign up/Login")
        print("3. Create an API key")
        print("4. Replace 'your_groq_api_key_here' in .env file with your actual key")
        return
    
    print("✅ Groq API key found!")
    
    # Initialize detector
    detector = ObjectDetector()
    
    # Test with a simple image (create a test frame)
    test_frame = cv2.imread("test_detection_result.jpg")
    
    if test_frame is None:
        print("❌ Test image not found. Please run 'python test_detection.py' first.")
        return
    
    print("📸 Testing AI analysis with sample image...")
    
    # Test the Groq analysis
    analysis = detector.analyze_frame_with_groq(test_frame)
    
    print("\n🤖 AI Analysis Result:")
    print("-" * 40)
    print(analysis)
    print("-" * 40)
    
    if "Error" in analysis or "API Error" in analysis:
        print("\n❌ API test failed. Check your API key and internet connection.")
    else:
        print("\n✅ Groq API integration working successfully!")

if __name__ == "__main__":
    test_groq_api()
