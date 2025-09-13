---
title: AI Plant Vision
emoji: 🏆
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.17.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: ' Plant Classifier AI - Integrated with Plant Info API '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# AI Plant Vision
https://huggingface.co/spaces/fadiyahalanazi/AI-plant-vision
## 🌿 Overview

AI Plant Vision is a Gradio-based web application that allows users to identify plants from images and retrieve detailed information about them in either English or Arabic. The system leverages a zero-shot image classification model and a text generation model to provide accurate plant identification and rich informational content.

## 🚀 Features

- Upload an image to identify a plant.
- Supports multiple plant types such as Lavender, Aloe Vera, Mint, and Orchids.
- Provides detailed plant information, including scientific name, growing conditions, common uses, and care tips.
- Supports English and Arabic languages.
- Uses advanced AI models for classification and text generation.
- Optimized for fast GPU-based inference to enhance performance.
- Modular pipeline for easy scalability and future model improvements.

## 🛠️ Installation & Setup

### Prerequisites

Ensure you have Python installed on your system (>=3.8).

### Step 1: Clone the Repository

```bash
git clone https://huggingface.co/spaces/fadiyahalanazi/AI-plant-vision
```

### Step 2: Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Step 3: Install Dependencies

The script automatically installs required packages. However, you can also install them manually:

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python app.py
```

## 📜 How It Works

1. Upload a plant image.
2. Choose your preferred language (English/Arabic).
3. Click the **"Identify & Get Info"** button.
4. The app classifies the plant and fetches detailed information.

## 📌 Technologies Used

- **Python** - Primary language for implementation.
- **Transformers** - Utilized for image classification and text generation.
- **Gradio** - Provides an interactive user interface.
- **Torch** - Facilitates deep learning model computations.
- **Hugging Face Spaces** - Used for deployment with GPU acceleration.

## 🖼️ User Interface

The app features a clean and responsive UI with:

- Custom styling.
- Intuitive controls for image input and language selection.
- Instant plant identification and information retrieval.

## 🔥 Model Details

- **Image Classification Model**: `umutbozdag/plant-identity`
- **Text Generation Models**:
  - English: `microsoft/Phi-3-mini-4k-instruct`
  - Arabic: `ALLaM-AI/ALLaM-7B-Instruct-preview`


