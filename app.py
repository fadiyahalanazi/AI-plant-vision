import os
import subprocess
import spaces

# Install necessary packages if not already installed
def install_packages():
    packages = ["transformers", "gradio", "requests", "torch"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.call(["pip", "install", package])

install_packages()  # Install dependencies before running the app

# Import required libraries
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import requests
import torch

# Load the image classification model
classifier = pipeline("image-classification", model="umutbozdag/plant-identity")


# Function to get the appropriate text-generation model
def get_model_name(language):
    model_mapping = {
        "English": "microsoft/Phi-3-mini-4k-instruct",
        "Arabic": "ALLaM-AI/ALLaM-7B-Instruct-preview"
    }
    return model_mapping.get(language, "ALLaM-AI/ALLaM-7B-Instruct-preview")

# Function to load the text-generation model
def load_text_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        do_sample=False
    )
    return generator
    
@spaces.GPU

# Function to classify plant images and fetch plant information
def classify_and_get_info(image, language):
    result = classifier(image)

    if result:  # Ensure result is not empty
        plant_name = result[0]["label"]  # Extract the top predicted class
    else:
        return "Unknown", "Could not classify the plant."


    # Load the appropriate text-generation model based on language
    model_name = get_model_name(language)
    text_generator = load_text_model(model_name)

    # Define the prompt for plant information
    prompt = (
        f"Provide detailed information about {plant_name}. Include its scientific name, growing conditions, common uses, and care tips."
        if language == "English"
        else f"قدم معلومات مفصلة عن {plant_name}. اذكر اسمه العلمي، وظروف نموه، واستخداماته الشائعة، ونصائح العناية به."
    )

    messages = [{"role": "user", "content": prompt}]
    output = text_generator(messages)

    plant_info = output[0]["generated_text"] if output else "No detailed information available."

    return plant_name, plant_info

# Gradio interface with a styled theme
with gr.Blocks(css="""
    .gradio-container { 
        background-color: #d9ccdf;
        font-family: 'Arial', sans-serif; 
        color: white;
        text-align: center;
    }
    h1 {
        color: #333333;
        font-size: 32px;
        margin-bottom: 10px;
    }
    p {
        color: #181817;
        font-size: 18px;
    }
    .gradio-container .btn {
        background-color: #00A86B !important;
        color: white !important;
        font-size: 18px;
        padding: 10px 20px;
        font-weight: bold;
        border-radius: 12px;
        border: none;
    }
    .gradio-container .btn:hover {
        background-color: #008554 !important;
    }
    .gradio-container .textbox {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        background-color: #b69dc2;
        border: 2px solid #00A86B;
        padding: 10px;
        border-radius: 8px;
    }
""") as interface:
    
    # App title and description
    gr.Markdown("<h1>🌿 AI Plant Visión </h1>")
    gr.Markdown("<p>Upload an image to identify a plant and retrieve detailed information in English or Arabic.</p>")

    # Layout for user input and results
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📸 Upload a Plant Image")
            language_selector = gr.Radio(["English", "Arabic"], label="🌍 Choose Language", value="English")
            classify_button = gr.Button("🔍 Identify & Get Info")
        with gr.Column(scale=1):
            plant_name_output = gr.Textbox(label="🌱 Identified Plant Name", interactive=False, elem_classes="textbox")
            plant_info_output = gr.Textbox(label="📖 Plant Information", interactive=False, lines=4, elem_classes="textbox")

    # Connect button click to function
    classify_button.click(classify_and_get_info, inputs=[image_input, language_selector], outputs=[plant_name_output, plant_info_output])

    # Footer
    gr.Markdown("<p style='font-size: 14px; text-align: center;'>Developed with ❤️ using Hugging Face & Gradio</p>")

# Launch the application with public link
interface.launch(share=True)