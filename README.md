# Real-Time Object Detection System with AI Analysis

A Python application that uses your laptop's webcam to detect objects in real-time, displaying bounding boxes and labels around detected objects. Now enhanced with **AI-powered scene analysis** using Groq's Llama-4-Scout model!

## Features

- **Real-time object detection** using YOLOv8 (nano model for speed)
- **80+ object classes** detection (person, car, bicycle, dog, cat, etc.)
- **Bounding boxes and labels** with confidence scores
- **🤖 AI Scene Analysis**: When a person is detected, automatically analyzes the scene using Groq's Llama-4-Scout model
- **FPS counter** for performance monitoring
- **Interactive controls**:
  - `q` - Quit the application
  - `c` - Toggle confidence threshold (0.3 ↔ 0.7)
  - `a` - Force AI analysis of current frame
  - `h` - Show help

## Requirements

- Python 3.8+
- Webcam/Camera
- Linux/Windows/macOS
- **Groq API Key** (for AI analysis features)

## Installation

1. **Clone or download** this project to your computer

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Get a Groq API Key** (for AI analysis):
   - Visit [https://console.groq.com/](https://console.groq.com/)
   - Sign up/Login
   - Create an API key
   - Copy your API key

5. **Configure environment variables**:
   - Open the `.env` file
   - Replace `your_groq_api_key_here` with your actual Groq API key:
   ```
   GROQ_API_KEY=your_actual_api_key_here
   ```

## Usage

### Quick Start

1. **Run the main application**:
   ```bash
   python main.py
   ```

2. **Camera window will open** showing real-time detection
3. **Press `q` to quit** when you're done

### Test the System

1. **Test basic detection** (without camera):
   ```bash
   python test_detection.py
   ```

2. **Test Groq AI integration**:
   ```bash
   python test_groq.py
   ```

These will:
- Test object detection using a sample image
- Check camera access
- Verify Groq API connectivity
- Provide troubleshooting information

## How It Works

1. **Model Loading**: Downloads YOLOv8 nano model (6.25MB) on first run
2. **Camera Access**: Opens your default webcam (usually camera index 0)
3. **Real-time Processing**: 
   - Captures frames from the camera
   - Runs YOLO inference on each frame
   - Draws bounding boxes and labels
   - **🤖 AI Analysis**: When a person is detected, sends frame to Groq API for intelligent analysis
   - Displays the result with AI insights

## AI Analysis Features

- **Automatic Triggering**: AI analysis runs automatically when a person is detected
- **Smart Cooldown**: Prevents API spam with 10-second intervals between analyses
- **Detailed Insights**: Describes what's happening in the scene, focusing on people and activities
- **Security Focus**: Identifies potential security concerns
- **Non-blocking**: Analysis runs in background without affecting real-time detection

## Detected Object Classes

The system can detect 80 different object classes, including:

- **People**: person
- **Vehicles**: car, motorcycle, airplane, bus, train, truck, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Objects**: bicycle, bottle, wine glass, cup, fork, knife, spoon, bowl
- **Electronics**: laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster
- **Furniture**: chair, couch, potted plant, bed, dining table, toilet
- **Sports**: frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
- And many more!

## Performance Tips

- **For better performance**: Use a smaller resolution (the script defaults to 640x480)
- **For higher accuracy**: Change model from `yolov8n.pt` to `yolov8s.pt` or `yolov8m.pt` (slower but more accurate)
- **GPU acceleration**: If you have CUDA-compatible GPU, PyTorch will automatically use it

## Troubleshooting

### Camera Issues

If camera doesn't work:

1. **Check if camera is available**:
   ```bash
   ls /dev/video*  # Linux
   ```

2. **Give camera permissions** (Linux):
   ```bash
   sudo chmod 666 /dev/video0
   ```

3. **Close other applications** that might be using the camera

4. **Try different camera index**:
   - Edit `main.py` and change `camera_index=0` to `camera_index=1` or `camera_index=2`

### Display Issues

If you get display-related errors:

1. **For Linux servers without display**:
   - Install X11 forwarding: `ssh -X username@server`
   - Or use VNC/remote desktop

2. **For headless systems**:
   - Use the test script to verify detection works
   - Modify the main script to save images instead of displaying

### Performance Issues

1. **Lower FPS**:
   - Close other applications
   - Use smaller resolution
   - Use CPU-only mode by setting `device='cpu'` in YOLO initialization

2. **High CPU usage**:
   - Normal for real-time detection
   - Consider using a faster computer or GPU acceleration

## Configuration

You can modify the following in `main.py`:

```python
# Model selection (nano is fastest, but less accurate)
model_name = 'yolov8n.pt'  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Confidence threshold
confidence_threshold = 0.5  # Range: 0.0 to 1.0

# Camera settings
camera_index = 0  # Try 1, 2, etc. if 0 doesn't work
frame_width = 640
frame_height = 480
fps = 30
```

## File Structure

```
ai_security/
├── main.py              # Main application with AI integration
├── test_detection.py    # Test basic object detection
├── test_groq.py         # Test Groq API integration
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (API keys)
├── README.md           # This file
├── myenv/              # Virtual environment
└── yolov8n.pt         # YOLO model (downloaded automatically)
```

## License

This project uses open-source libraries:
- **OpenCV**: Apache 2.0 License
- **Ultralytics YOLO**: AGPL-3.0 License
- **PyTorch**: BSD License

## Credits

- **YOLO**: Ultralytics YOLOv8
- **Computer Vision**: OpenCV
- **Deep Learning**: PyTorch

---

**Enjoy real-time object detection! 🎯📹**
