import torch
import torch.nn as nn
import numpy as np
import base64
from flask import Flask, render_template, jsonify
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# --- Model Configuration ---
LATENT_DIM = 100
IMG_SIZE = 784
MODEL_PATH = "models/generator_latest.pth"
DEVICE = torch.device("cpu")

# --- Generator class (must match the one in training script) ---
class Generator(nn.Module):
    def __init__(self, latent_dim, img_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# --- Load Model ---
generator = Generator(LATENT_DIM, IMG_SIZE).to(DEVICE)
try:
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    generator.eval()
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    try:
        # Generate random noise
        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        
        # Generate image
        with torch.no_grad():
            gen_img = generator(z).cpu().numpy()
        
        # Rescale to [0, 255]
        gen_img = (0.5 * gen_img + 0.5) * 255
        gen_img = gen_img.astype(np.uint8).reshape(28, 28)
        
        # Convert to PIL and then to base64
        pil_img = Image.fromarray(gen_img)
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({'image': img_str})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
