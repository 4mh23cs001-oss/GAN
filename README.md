# MNIST GAN: Handwritten Digit Generator 🎨

This project implements a **Generative Adversarial Network (GAN)** using PyTorch to generate realistic handwritten digits based on the MNIST dataset. It includes a complete training pipeline, pre-trained model checkpoints, and a Flask-based web interface for real-time digit generation.

## 🚀 Features

- **End-to-End GAN Training**: Custom training loop for both Generator and Discriminator.
- **Pre-trained Models**: Includes checkpoints for various training epochs.
- **Web Interface**: A premium Flask-based UI to generate and visualize digits instantly.
- **Progress Tracking**: Automatic saving of sample images during training to monitor quality.

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch
- **Backend**: Flask (Python)
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Frontend**: HTML5, CSS3 (Vanilla)

## 📁 Project Structure

```text
.
├── app.py              # Flask Web Application
├── mnist_gan.py        # GAN Training Script & Model Architecture
├── models/             # Saved PyTorch model checkpoints (.pth)
├── samples/            # Generated samples throughout training epochs
├── templates/          # HTML templates for the web UI
└── .gitignore          # Git exclusion rules
```

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/4mh23cs001-oss/GAN.git
   cd GAN
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision pandas numpy matplotlib flask
   ```

3. **Required Dataset**:
   Ensure you have the `mnist_dataset.csv` file in the root directory (or download it from Kaggle/MNIST).

## 🎮 Usage

### 1. Training the GAN
To start training the model from scratch:
```bash
python mnist_gan.py
```
*Models will be saved in `/models` and visual samples in `/samples` every 2 epochs.*

### 2. Running the Web App
To launch the interactive generator UI:
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser to start generating digits!

## 📊 GAN Architecture

- **Generator**: 4 fully connected layers with LeakyReLU activation and Tanh output.
- **Discriminator**: 3 fully connected layers with LeakyReLU and Sigmoid output for binary classification.
- **Latent Space**: 100-dimensional random noise vector.

## ✨ Results
Below is an example of the progression of generated digits after 100 epochs of training:
*(Check the `samples/` folder for visual results from Epoch 2 to 100)*

---
Developed by [KEERTHANA](https://github.com/4mh23cs001-oss)
