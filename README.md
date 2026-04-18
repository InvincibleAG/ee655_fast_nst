# 🎨 Neural Style Transfer - Full Stack Application

A modern, full-stack web application for neural style transfer. Upload content and style images to generate beautifully stylized artwork using VGG19 deep neural networks and PyTorch.

## ✨ Features

- **Real-time Style Transfer** - Transform images with artistic styles
- **3 Processing Modes**:
  - ⚡ **Fast Mode** - Quick results, lower quality (best for previews)
  - ⚖️ **Balanced Mode** - Recommended, good speed/quality trade-off
  - 🎯 **Quality Mode** - Best results, slower processing
- **Flexible Resolution** - Support for 128x128, 256x256, 512x512
- **Performance Metrics** - View processing time and loss values
- **Download Results** - Save your stylized images as PNG
- **Modern UI** - Beautiful, responsive React interface
- **Docker Support** - Easy deployment with Docker/Docker Compose

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
quickstart.bat
```

**macOS/Linux:**
```bash
chmod +x quickstart.sh
./quickstart.sh
```

### Option 2: Manual Setup

**1. Backend Setup:**
```bash
# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-backend.txt

# Run backend
python app.py
```

**2. Frontend Setup (in another terminal):**
```bash
cd frontend
npm install
npm start
```

**3. Open in Browser:**
```
http://localhost:3000
```

### Option 3: Docker

```bash
docker-compose up
```

Then open http://localhost:3000

## 📋 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|------------|
| Python | 3.8 | 3.10+ |
| Node.js | 14 | 18+ |
| RAM | 4GB | 8GB+ |
| Storage | 2GB | 5GB+ |
| GPU | Optional | CUDA 11.8+ |

## 📚 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed installation and configuration
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation

## 🎯 How to Use

1. **Upload Content Image** - Your base photo
2. **Upload Style Image** - Reference artwork or photo
3. **Configure Settings**:
   - Select processing mode
   - Adjust number of steps (more = better quality)
   - Choose resolution
4. **Click Stylize** - Let the AI work its magic
5. **Download Result** - Save your stylized image

## 🏗️ Architecture

### Backend Stack
- **Framework**: Flask with CORS support
- **ML**: PyTorch + TorchVision
- **Model**: VGG19 (pre-trained)
- **Image Processing**: Pillow, OpenCV
- **Loss Functions**: Content Loss, Style Loss (Gram matrices)

### Frontend Stack
- **Framework**: React 18
- **HTTP Client**: Axios
- **Styling**: CSS3 with animations
- **State Management**: React Hooks

## 📁 Project Structure

```
Neural-Style-Transfer/
├── app.py                              # Flask backend server
├── requirements-backend.txt            # Python dependencies
├── SETUP_GUIDE.md                      # Setup instructions
├── API_REFERENCE.md                    # API documentation
├── Dockerfile.backend                  # Backend container
├── docker-compose.yml                  # Docker orchestration
├── quickstart.sh                       # Unix quick start
├── quickstart.bat                      # Windows quick start
│
├── frontend/
│   ├── package.json                   # Node dependencies
│   ├── .env                           # Environment config
│   ├── Dockerfile                     # Frontend container
│   │
│   ├── public/
│   │   └── index.html                # HTML entry point
│   │
│   └── src/
│       ├── index.js                  # React entry point
│       ├── index.css                 # Global styles
│       ├── App.js                    # Main component
│       └── App.css                   # Component styles
│
└── notebooks/
    ├── Fast_Neural_Style_Transfer.ipynb    # Jupyter notebook
    └── new.py                         # Utilities module
```

## 🔄 Processing Modes Comparison

| Feature | Fast | Balanced | Quality |
|---------|------|----------|---------|
| Content Layers | conv_2 | conv_4 | conv_4 |
| Style Layers | conv_1,2 | conv_1-5 | conv_1-5 |
| Default Steps | ≤50 | 100-200 | ≥200 |
| Processing Time | ~10-30s | ~30-60s | ~60-300s |
| Quality | Good | Excellent | Outstanding |
| Best For | Previews | General Use | Final Output |

## 🔌 API Endpoints

### Health Check
```http
GET /health
```

### Style Transfer
```http
POST /api/stylize
Content-Type: multipart/form-data

Parameters:
- content: File (image/jpg or image/png)
- style: File (image/jpg or image/png)
- num_steps: Number (default: 100)
- size: Number (128, 256, or 512; default: 256)
- mode: String ('fast', 'balanced', or 'quality'; default: 'balanced')
```

## 💡 Tips for Best Results

1. **Image Quality** - Use high-resolution input images (512x512+)
2. **Style Reference** - Choose distinctive artworks or artistic photos
3. **Content Preservation** - Use balanced mode to keep original content
4. **Processing Time** - Start with 256x256 at balanced mode
5. **Experimentation** - Try different style images and settings
6. **Batch Processing** - Process multiple images with same style for consistency

## 🐛 Troubleshooting

### "Error connecting to server"
- Backend not running? Start it: `python app.py`
- Wrong port? Check port 5000 is available
- Firewall issue? Check firewall settings

### "Invalid file type"
- Only JPG and PNG supported
- Convert other formats to PNG first

### "Out of Memory"
- Reduce image size to 128x128
- Switch to Fast mode
- Reduce number of steps
- Close other applications

### "Slow Processing"
- Reduce image resolution
- Use Fast mode
- Decrease number of steps
- Enable GPU if available (CUDA)

## ⚙️ Configuration

### Backend Configuration (`app.py`)
```python
UPLOAD_FOLDER = 'uploads'          # Upload directory
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10MB
```

### Frontend Configuration (`frontend/.env`)
```
REACT_APP_API_URL=http://localhost:5000
```

### GPU Support
To use CUDA (requires NVIDIA GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🐳 Docker Deployment

### Build and Run
```bash
docker-compose up --build
```

### Services
- **Backend**: http://localhost:5000
- **Frontend**: http://localhost:3000

### Environment Variables
Set in `docker-compose.yml` if needed:
- `FLASK_ENV` - Set to 'production' for deployment
- `REACT_APP_API_URL` - API endpoint (usually `http://backend:5000`)

## 📊 Performance Benchmarks

*Approximate times on CPU (Core i7, 8GB RAM):*

| Resolution | Steps | Mode | Time |
|------------|-------|------|------|
| 128x128 | 50 | Fast | 5-10s |
| 256x256 | 100 | Balanced | 30-45s |
| 256x256 | 200 | Quality | 60-90s |
| 512x512 | 100 | Balanced | 120-180s |

*GPU speeds can be 5-10x faster*

## 🎓 Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) - Original NST Paper
- [VGG19 Paper](https://arxiv.org/abs/1409.1556)
- [TorchVision Documentation](https://pytorch.org/vision/stable/)
- [Gram Matrices Explained](https://en.wikipedia.org/wiki/Gramian_matrix)

## 🛠️ Development

### Adding Features

1. **New Style Transfer Modes** - Edit `app.py` in the `/api/stylize` route
2. **UI Improvements** - Modify `frontend/src/App.js` and `.css`
3. **Performance Optimization** - Adjust model layers or parameters

### Running Tests
```bash
# Backend tests (create tests/ directory)
pytest tests/

# Frontend tests
cd frontend
npm test
```

## 📦 Production Deployment

### Backend (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Frontend (Build)
```bash
cd frontend
npm run build
# Deploy build/ directory to static hosting
```

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Add support for more image formats
- [ ] Implement real-time preview
- [ ] Add batch processing
- [ ] Optimize memory usage
- [ ] Add more style transfer algorithms
- [ ] Improve error handling

## 📞 Support

### Getting Help
1. Check [SETUP_GUIDE.md](SETUP_GUIDE.md)
2. Review [API_REFERENCE.md](API_REFERENCE.md)
3. Check troubleshooting section above
4. Examine console logs and terminal output

### Common Issues

**"ModuleNotFoundError: No module named..."**
- Run: `pip install -r requirements-backend.txt`

**"npm: command not found"**
- Install Node.js from https://nodejs.org/

**Port already in use**
- Backend: Change port in `app.py` `app.run(port=5001)`
- Frontend: Set `PORT=3001 npm start`

## 🎉 Examples

### Artistic Styles
- Oil paintings
- Watercolor effects
- Sketches and drawings
- Famous artwork styles (Van Gogh, Picasso, etc.)

### Content Photos
- Portraits
- Landscapes
- Architecture
- Still life

## 📈 What's Next?

Possible enhancements:
- [ ] Pre-trained fast style transfer models
- [ ] Real-time webcam style transfer
- [ ] Multiple style blending
- [ ] Advanced color control
- [ ] Mobile app version
- [ ] Cloud deployment

---

<div align="center">

**Made with ❤️ for style transfer enthusiasts**

[⬆ Back to Top](#-neural-style-transfer---full-stack-application)

</div>
