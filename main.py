import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import time
import copy
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(path, size=512):
    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def save_image(tensor, path):
    image = tensor.cpu().clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)

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

def run_style_transfer(content_img, style_img,
                       content_layers, style_layers,
                       num_steps=300,
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

if __name__ == "__main__":

    os.makedirs("outputs", exist_ok=True)

    content_path = "content.jpg"
    style_path = "style.jpg"

    content_full = load_image(content_path, size=512)
    style_full = load_image(style_path, size=512)

    full_content_layers = ['conv_4']
    full_style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    print("Running FULL NST...")
    out_full, time_full, s_loss_full, c_loss_full = run_style_transfer(
        content_full, style_full,
        full_content_layers, full_style_layers,
        num_steps=300
    )

    save_image(out_full, "outputs/full_result.jpg")

    content_fast = load_image(content_path, size=256)  # smaller resolution
    style_fast = load_image(style_path, size=256)

    fast_content_layers = ['conv_2']
    fast_style_layers = ['conv_1', 'conv_2']

    print("Running FAST NST...")
    out_fast, time_fast, s_loss_fast, c_loss_fast = run_style_transfer(
        content_fast, style_fast,
        fast_content_layers, fast_style_layers,
        num_steps=100  # fewer iterations
    )

    save_image(out_fast, "outputs/fast_result.jpg")

    print("\n===== PERFORMANCE =====")
    print(f"Full Model Time: {time_full:.2f}s")
    print(f"Fast Model Time: {time_fast:.2f}s")

    print("\n===== LOSSES =====")
    print(f"Full Style Loss: {s_loss_full:.4f}")
    print(f"Fast Style Loss: {s_loss_fast:.4f}")

    print(f"Full Content Loss: {c_loss_full:.4f}")
    print(f"Fast Content Loss: {c_loss_fast:.4f}")

