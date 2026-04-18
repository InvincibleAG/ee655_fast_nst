from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import io
import os
import base64
import time
import copy
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================
# Utility Functions
# ========================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image_from_bytes(file_bytes, size=256):
    """Load image from bytes and convert to tensor"""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image_tensor = loader(image).unsqueeze(0)
    return image_tensor.to(device, torch.float)

def tensor_to_base64(tensor):
    """Convert tensor to base64 string"""
    image = tensor.cpu().clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)

    # Convert to bytes
    img_io = io.BytesIO()
    image.save(img_io, 'PNG', quality=95)
    img_io.seek(0)

    # Convert to base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# ========================
# Loss Modules
# ========================

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# ========================
# Model Builder
# ========================

def get_model_and_losses(cnn, style_img, content_img,
                         content_layers, style_layers):
    cnn = copy.deepcopy(cnn)
    content_losses = []
    style_losses = []
    model = nn.Sequential()

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses

# ========================
# Style Transfer
# ========================

def run_style_transfer(content_img, style_img,
                       content_layers, style_layers,
                       num_steps=100,
                       style_weight=1e6,
                       content_weight=1):
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    model, style_losses, content_losses = get_model_and_losses(
        cnn, style_img, content_img,
        content_layers, style_layers
    )

    input_img = content_img.clone()
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    start_time = time.time()

    while run[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            model(input_img)

            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            run[0] += 1
            return loss

        optimizer.step(closure)

    end_time = time.time()

    final_style_loss = sum(sl.loss.item() for sl in style_losses)
    final_content_loss = sum(cl.loss.item() for cl in content_losses)

    return input_img, end_time - start_time, final_style_loss, final_content_loss

# ========================
# Routes
# ========================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'device': str(device)})

@app.route('/api/stylize', methods=['POST'])
def stylize():
    try:
        # Check if files are present
        if 'content' not in request.files or 'style' not in request.files:
            return jsonify({'error': 'Missing content or style image'}), 400

        content_file = request.files['content']
        style_file = request.files['style']

        # Check file names
        if content_file.filename == '' or style_file.filename == '':
            return jsonify({'error': 'No selected files'}), 400

        # Check file extensions
        if not (allowed_file(content_file.filename) and allowed_file(style_file.filename)):
            return jsonify({'error': 'Invalid file type. Only JPG and PNG allowed'}), 400

        # Get parameters
        num_steps = int(request.form.get('num_steps', 100))
        size = int(request.form.get('size', 256))
        mode = request.form.get('mode', 'balanced')  # 'fast', 'balanced', 'quality'

        # Adjust parameters based on mode
        if mode == 'fast':
            content_layers = ['conv_2']
            style_layers = ['conv_1', 'conv_2']
            num_steps = min(num_steps, 50)
        elif mode == 'quality':
            content_layers = ['conv_4']
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
            num_steps = max(num_steps, 200)
        else:  # balanced
            content_layers = ['conv_4']
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        # Load images
        content_bytes = content_file.read()
        style_bytes = style_file.read()

        content_img = load_image_from_bytes(content_bytes, size=size)
        style_img = load_image_from_bytes(style_bytes, size=size)

        # Run style transfer
        print(f"Starting style transfer: {mode} mode, {num_steps} steps, size {size}x{size}")
        output_img, elapsed_time, style_loss, content_loss = run_style_transfer(
            content_img, style_img,
            content_layers, style_layers,
            num_steps=num_steps
        )

        # Convert to base64
        output_base64 = tensor_to_base64(output_img)

        return jsonify({
            'success': True,
            'output': output_base64,
            'time': f"{elapsed_time:.2f}s",
            'style_loss': f"{style_loss:.4f}",
            'content_loss': f"{content_loss:.4f}"
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
