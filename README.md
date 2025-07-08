# YOLO Object Detection and Tracking

A comprehensive Python script for real-time object detection and tracking using Ultralytics YOLO models with support for multiple input sources and hardware acceleration.

## Features

- 🚀 **Multiple YOLO Models**: Support for YOLOv5, YOLOv8, YOLO11, and YOLO12
- 📹 **Flexible Input Sources**: Camera, video files, RTSP/HTTP streams
- 🎯 **Advanced Tracking**: ByteTrack and BoT-SORT integration
- ⚡ **Hardware Acceleration**: CPU, NVIDIA GPU (CUDA), Apple GPU (MPS)
- 🎛️ **FPS Control**: Decouple inference FPS from video reading
- 📊 **Rich Visualization**: Bounding boxes, labels, confidence scores, tracking trails
- 💾 **Output Saving**: Save processed videos with annotations
- 🔧 **Easy Configuration**: Command-line interface with sensible defaults

## Installation

### Step 1: Clone or Download

```bash
# If using git
git clone <repository-url>
cd yolo-tracking

# Or download the files directly
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv yolo-env

# Activate virtual environment
# On Windows:
yolo-env\Scripts\activate
# On macOS/Linux:
source yolo-env/bin/activate
```

### Step 3: Install Dependencies

#### Option A: Using requirements.txt (Recommended)

```bash
pip install -r requirements.txt
```

#### Option B: Manual Installation

```bash
# Core dependencies
pip install ultralytics>=8.0.0
pip install opencv-python>=4.5.0
pip install numpy>=1.21.0
pip install torch>=1.9.0
pip install torchvision>=0.10.0
pip install Pillow>=8.3.0

# Additional dependencies
pip install matplotlib seaborn pandas scipy
pip install imageio imageio-ffmpeg
pip install psutil tqdm pyyaml
```

### Step 4: GPU Support (Optional but Recommended)

#### NVIDIA GPU (CUDA)
```bash
# Uninstall CPU version first
pip uninstall torch torchvision torchaudio

# Install CUDA version (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Apple Silicon (M1/M2/M3)
```bash
# PyTorch with MPS support is included in the default installation
# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Step 5: Verify Installation

```bash
# Test the installation
python yolo_tracker.py --list-models

# Quick test with default camera
python yolo_tracker.py --source 0 --inference-fps 5
```

### System Requirements

#### Minimum Requirements
- Python 3.8 or higher
- 4GB RAM
- 2GB free disk space
- Webcam or video files for testing

#### Recommended Requirements
- Python 3.9 or higher
- 8GB+ RAM
- GPU with 4GB+ VRAM (NVIDIA) or Apple Silicon
- 5GB+ free disk space (for models and outputs)

### Troubleshooting Installation

#### Common Issues

1. **Python version compatibility**:
   ```bash
   python --version  # Should be 3.8+
   ```

2. **Permission errors**:
   ```bash
   # Use --user flag if needed
   pip install --user -r requirements.txt
   ```

3. **CUDA installation issues**:
   ```bash
   # Check NVIDIA driver
   nvidia-smi
   
   # Check CUDA toolkit version
   nvcc --version
   ```

4. **Package conflicts**:
   ```bash
   # Create fresh environment
   conda create -n yolo python=3.9
   conda activate yolo
   pip install -r requirements.txt
   ```

## Quick Start

```bash
# Basic usage with default camera
python yolo_tracker.py

# Process a video file
python yolo_tracker.py --source video.mp4

# Use NVIDIA GPU acceleration
python yolo_tracker.py --device cuda --model yolo11s.pt
```

## Available Models

### YOLOv5 Models
- `yolov5n.pt` - Nano (fastest, least accurate)
- `yolov5s.pt` - Small
- `yolov5m.pt` - Medium
- `yolov5l.pt` - Large
- `yolov5x.pt` - Extra Large (slowest, most accurate)

### YOLOv8 Models
- `yolov8n.pt` - Nano
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large

### YOLO11 Models (Latest)
- `yolo11n.pt` - Nano
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large

### YOLO12 Models (Newest)
- `yolo12n.pt` - Nano
- `yolo12s.pt` - Small
- `yolo12m.pt` - Medium
- `yolo12l.pt` - Large
- `yolo12x.pt` - Extra Large

## Usage

### Command Line Arguments

```bash
python yolo_tracker.py [OPTIONS]
```

#### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | Video source (0 for camera, file path, or RTSP URL) |
| `--model` | `yolo11n.pt` | YOLO model name |
| `--device` | `auto` | Device to use (auto, cpu, cuda, mps) |
| `--tracker` | `bytetrack` | Tracker type (bytetrack, botsort, deepsort) |
| `--inference-fps` | `30` | FPS for inference processing |
| `--mode` | `track` | Mode (detect, track) |
| `--save` | `False` | Save output video |
| `--output` | `output.mp4` | Output video path |
| `--list-models` | `False` | List available YOLO models |

### Usage Examples

#### Basic Usage

```bash
# Run with default settings (camera, YOLO11n, ByteTrack, auto device)
python yolo_tracker.py

# List available models
python yolo_tracker.py --list-models
```

#### Camera Input

```bash
# Use default camera (index 0)
python yolo_tracker.py --source 0

# Use specific camera (index 1)
python yolo_tracker.py --source 1

# Use external USB camera
python yolo_tracker.py --source 2
```

#### Video File Input

```bash
# Process video file
python yolo_tracker.py --source path/to/video.mp4

# Process video with different model
python yolo_tracker.py --source video.mp4 --model yolov8s.pt

# Process and save output
python yolo_tracker.py --source input.mp4 --save --output processed.mp4
```

#### RTSP/HTTP Stream Input

```bash
# Process RTSP stream
python yolo_tracker.py --source rtsp://username:password@192.168.1.100:554/stream

# Process HTTP stream
python yolo_tracker.py --source http://192.168.1.100:8080/stream

# Process IP camera with authentication
python yolo_tracker.py --source rtsp://admin:password@camera-ip:554/h264
```

#### Model Selection

```bash
# Use different YOLO models
python yolo_tracker.py --model yolov5n.pt    # Fastest
python yolo_tracker.py --model yolov8s.pt    # Balanced
python yolo_tracker.py --model yolo11m.pt    # More accurate
python yolo_tracker.py --model yolo12l.pt    # Highest accuracy
```

#### Device Selection

```bash
# Auto-detect best available device (default)
python yolo_tracker.py --device auto

# Force CPU usage
python yolo_tracker.py --device cpu

# Use NVIDIA GPU (CUDA)
python yolo_tracker.py --device cuda

# Use Apple GPU (Metal Performance Shaders)
python yolo_tracker.py --device mps
```

#### Tracker Selection

```bash
# Use ByteTrack (default, recommended)
python yolo_tracker.py --tracker bytetrack

# Use BoT-SORT
python yolo_tracker.py --tracker botsort

# Use DeepSORT (falls back to detection for now)
python yolo_tracker.py --tracker deepsort
```

#### Inference FPS Control

```bash
# High performance (process more frames)
python yolo_tracker.py --inference-fps 30

# Balanced performance
python yolo_tracker.py --inference-fps 15

# Low resource usage
python yolo_tracker.py --inference-fps 5

# High-end GPU setup
python yolo_tracker.py --inference-fps 60
```

#### Mode Selection

```bash
# Detection only (no tracking)
python yolo_tracker.py --mode detect

# Detection with tracking (default)
python yolo_tracker.py --mode track
```

#### Advanced Examples

```bash
# High-performance setup with NVIDIA GPU
python yolo_tracker.py \
    --source 0 \
    --model yolo11s.pt \
    --device cuda \
    --tracker bytetrack \
    --inference-fps 30 \
    --save \
    --output high_perf.mp4

# Apple Silicon optimized
python yolo_tracker.py \
    --source video.mp4 \
    --model yolo11n.pt \
    --device mps \
    --tracker bytetrack \
    --inference-fps 25

# RTSP stream processing with save
python yolo_tracker.py \
    --source rtsp://admin:password@192.168.1.100:554/stream \
    --model yolo11m.pt \
    --device cuda \
    --tracker bytetrack \
    --inference-fps 20 \
    --save \
    --output rtsp_tracking.mp4

# CPU-optimized for low-end hardware
python yolo_tracker.py \
    --source webcam.mp4 \
    --model yolov5n.pt \
    --device cpu \
    --tracker bytetrack \
    --inference-fps 10
```

## Performance Optimization

### Hardware Recommendations

#### NVIDIA GPU (CUDA)
- **Minimum**: GTX 1060 6GB or better
- **Recommended**: RTX 30/40 series
- **Models**: Use medium to large models
- **FPS**: 20-60 FPS depending on model size

#### Apple Silicon (M1/M2/M3)
- **Minimum**: M1 with 8GB+ unified memory
- **Recommended**: M2/M3 with 16GB+ memory
- **Models**: Use nano to small models for optimal performance
- **FPS**: 15-30 FPS typically achievable

#### CPU Only
- **Minimum**: Intel i5/i7 or AMD Ryzen 5/7
- **Recommended**: Latest generation processors
- **Models**: Use nano models for real-time performance
- **FPS**: 5-15 FPS on most systems

### Performance Tips

1. **Model Selection**:
   - Use smaller models (nano/small) for real-time applications
   - Use larger models (medium/large/extra-large) for accuracy
   - YOLO11 and YOLO12 offer the best speed/accuracy balance

2. **Device Optimization**:
   - Use `--device auto` for automatic best device selection
   - Monitor GPU memory usage with larger models
   - Reduce inference FPS if experiencing memory issues

3. **FPS Configuration**:
   - **NVIDIA GPU**: 20-60 FPS depending on model
   - **Apple Silicon**: 15-30 FPS
   - **CPU**: 5-15 FPS
   - For RTSP streams, use lower inference FPS to reduce network load

4. **Memory Management**:
   - Close other applications when using GPU acceleration
   - Use smaller models if running out of VRAM
   - Consider batch processing for multiple streams

## Tracking Features

### Visual Elements
- **Bounding Boxes**: Color-coded by object class
- **Labels**: Object class, tracking ID, confidence score
- **Tracking Trails**: Visual path showing object movement history
- **Real-time Info**: Model, device, tracker, and FPS information

### Tracking Capabilities
- **Multi-object Tracking**: Track multiple objects simultaneously
- **ID Persistence**: Maintain object IDs across frames
- **Occlusion Handling**: Robust tracking through temporary occlusions
- **Real-time Performance**: Optimized for live video processing

## Controls

- **Quit**: Press `q` to exit the application
- **Stop**: Press `Ctrl+C` to stop processing
- **Display**: Real-time visualization window shows processed frames

## Troubleshooting

### Common Issues

1. **CUDA not available**:
   ```bash
   # Check if CUDA is properly installed
   python -c "import torch; print(torch.cuda.is_available())"
   # If False, install CUDA version of PyTorch
   ```

2. **MPS not available**:
   ```bash
   # Check if MPS is available (Apple Silicon only)
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

3. **Model download fails**:
   - Check internet connection
   - Models are downloaded automatically on first use
   - Download location: `~/.ultralytics/`

4. **Low FPS performance**:
   - Reduce inference FPS: `--inference-fps 10`
   - Use smaller model: `--model yolov5n.pt`
   - Check device usage: `--device auto`

5. **RTSP stream issues**:
   - Verify stream URL and credentials
   - Check network connectivity
   - Try lower inference FPS for network streams

### Debug Information

The script provides detailed information about:
- Selected device and availability
- Model loading status
- Video source properties
- Real-time processing statistics

## File Structure

```
yolo-tracking/
├── yolo_tracker.py          # Main script
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── outputs/               # Output videos (created automatically)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLO implementation
- ByteTrack and BoT-SORT for tracking algorithms
- OpenCV for video processing
- PyTorch for deep learning framework

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Ultralytics documentation
3. Create an issue in the repository

## Version History

- **v1.0.0**: Initial release with basic detection and tracking
- **v1.1.0**: Added GPU support (CUDA, MPS)
- **v1.2.0**: Enhanced tracking with ByteTrack integration
- **v1.3.0**: Added YOLO11 and YOLO12 model support
