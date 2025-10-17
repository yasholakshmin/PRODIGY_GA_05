import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Load image (small) ----
def load_image(path, max_size=256):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0))
    ])
    return transform(image).to(device)

def save_image(tensor, filename="output"):
    image = tensor.cpu().clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(f"{filename}.png")
    print(f"Saved image: {filename}.png")

# Load content and style
content = load_image("my_photo.jpg")
style = load_image("starry_night.jpg")

save_image(content, "content_image")
save_image(style, "style_image")

# ---- Load VGG19 ----
weights = VGG19_Weights.DEFAULT
vgg = vgg19(weights=weights).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# ---- Feature extraction ----
def get_features(image, model):
    layers = {'0':'conv1_1','5':'conv2_1','10':'conv3_1','19':'conv4_1','21':'conv4_2','28':'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, n_filters, h, w = tensor.size()
    tensor = tensor.view(n_filters, h*w)
    return torch.mm(tensor, tensor.t())

content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# ---- Initialize target ----
target = content.clone().requires_grad_(True).to(device)

# ---- Weights and optimizer ----
style_weights = {'conv1_1':1.0,'conv2_1':0.8,'conv3_1':0.5,'conv4_1':0.3,'conv5_1':0.1}
content_weight = 1e4
style_weight = 1e2

optimizer = optim.Adam([target], lr=0.003)

# ---- Fast style transfer loop ----
steps = 50  # small number of steps
for i in range(steps):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    style_loss = 0
    for layer in style_weights:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        style_loss += style_weights[layer]*torch.mean((target_gram - style_gram)**2)
    
    total_loss = content_weight*content_loss + style_weight*style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Step {i}, Total Loss: {total_loss.item():.2f}")

# ---- Save final image ----
save_image(target, "stylized_output_fast")
print("✅ Fast style transfer complete!")


