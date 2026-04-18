import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [contentImage, setContentImage] = useState(null);
  const [styleImage, setStyleImage] = useState(null);
  const [contentPreview, setContentPreview] = useState(null);
  const [stylePreview, setStylePreview] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [numSteps, setNumSteps] = useState(100);
  const [imageSize, setImageSize] = useState(256);
  const [mode, setMode] = useState('balanced');
  const [metrics, setMetrics] = useState(null);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  const handleImageChange = (e, type) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      if (type === 'content') {
        setContentImage(file);
        setContentPreview(event.target.result);
      } else {
        setStyleImage(file);
        setStylePreview(event.target.result);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleStylize = async () => {
    if (!contentImage || !styleImage) {
      setError('Please upload both content and style images');
      return;
    }

    const formData = new FormData();
    formData.append('content', contentImage);
    formData.append('style', styleImage);
    formData.append('num_steps', numSteps);
    formData.append('size', imageSize);
    formData.append('mode', mode);

    setLoading(true);
    setError(null);
    setOutputImage(null);

    try {
      const response = await axios.post(`${API_URL}/api/stylize`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setOutputImage(response.data.output);
        setMetrics({
          time: response.data.time,
          styleLoss: response.data.style_loss,
          contentLoss: response.data.content_loss,
        });
      } else {
        setError(response.data.error || 'Style transfer failed');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Error connecting to server');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setContentImage(null);
    setStyleImage(null);
    setContentPreview(null);
    setStylePreview(null);
    setOutputImage(null);
    setError(null);
    setMetrics(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎨 Neural Style Transfer</h1>
        <p>Transform your images with AI-powered style transfer</p>
      </header>

      <div className="app-container">
        <div className="control-panel">
          <h2>Settings</h2>

          <div className="setting-group">
            <label>Mode:</label>
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  value="fast"
                  checked={mode === 'fast'}
                  onChange={(e) => setMode(e.target.value)}
                />
                Fast (Quick, lower quality)
              </label>
              <label>
                <input
                  type="radio"
                  value="balanced"
                  checked={mode === 'balanced'}
                  onChange={(e) => setMode(e.target.value)}
                />
                Balanced (Recommended)
              </label>
              <label>
                <input
                  type="radio"
                  value="quality"
                  checked={mode === 'quality'}
                  onChange={(e) => setMode(e.target.value)}
                />
                Quality (Slow, best result)
              </label>
            </div>
          </div>

          <div className="setting-group">
            <label>Number of Steps: {numSteps}</label>
            <input
              type="range"
              min="10"
              max="300"
              step="10"
              value={numSteps}
              onChange={(e) => setNumSteps(parseInt(e.target.value))}
            />
            <small>More steps = better quality (slower)</small>
          </div>

          <div className="setting-group">
            <label>Image Size: {imageSize}x{imageSize}</label>
            <select
              value={imageSize}
              onChange={(e) => setImageSize(parseInt(e.target.value))}
            >
              <option value={128}>128x128 (Fastest)</option>
              <option value={256}>256x256 (Fast)</option>
              <option value={512}>512x512 (Slow)</option>
            </select>
          </div>
        </div>

        <div className="main-content">
          <div className="upload-section">
            <div className="image-upload">
              <h3>Content Image</h3>
              <div className="upload-box">
                {contentPreview ? (
                  <img src={contentPreview} alt="Content preview" className="preview" />
                ) : (
                  <div className="upload-placeholder">
                    <span>📷</span>
                    <p>Click to upload</p>
                  </div>
                )}
                <input
                  type="file"
                  accept="image/jpg,image/jpeg,image/png"
                  onChange={(e) => handleImageChange(e, 'content')}
                  className="file-input"
                />
              </div>
            </div>

            <div className="image-upload">
              <h3>Style Image</h3>
              <div className="upload-box">
                {stylePreview ? (
                  <img src={stylePreview} alt="Style preview" className="preview" />
                ) : (
                  <div className="upload-placeholder">
                    <span>🎨</span>
                    <p>Click to upload</p>
                  </div>
                )}
                <input
                  type="file"
                  accept="image/jpg,image/jpeg,image/png"
                  onChange={(e) => handleImageChange(e, 'style')}
                  className="file-input"
                />
              </div>
            </div>
          </div>

          <div className="button-group">
            <button
              className="btn btn-primary"
              onClick={handleStylize}
              disabled={loading || !contentImage || !styleImage}
            >
              {loading ? '✨ Processing...' : '✨ Stylize'}
            </button>
            <button className="btn btn-secondary" onClick={handleReset}>
              Reset
            </button>
          </div>

          {error && <div className="error-message">{error}</div>}

          {outputImage && (
            <div className="result-section">
              <h3>Generated Image</h3>
              <img src={outputImage} alt="Stylized output" className="output-image" />
              {metrics && (
                <div className="metrics">
                  <p>⏱️ Time: {metrics.time}</p>
                  <p>📊 Style Loss: {metrics.styleLoss}</p>
                  <p>📊 Content Loss: {metrics.contentLoss}</p>
                </div>
              )}
              <a
                href={outputImage}
                download="stylized.png"
                className="btn btn-download"
              >
                Download Result
              </a>
            </div>
          )}

          {loading && <div className="loading-spinner"></div>}
        </div>
      </div>

      <footer className="app-footer">
        <p>Powered by PyTorch & VGG19 • Neural Style Transfer</p>
      </footer>
    </div>
  );
}

export default App;
