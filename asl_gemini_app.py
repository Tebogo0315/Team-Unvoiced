

!pip install google-genai
!pip install google-generativeai gradio numpy opencv-python

# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import gradio as gr
import google.generativeai as genai
import numpy as np
import cv2

# Configure Gemini API
api_key = "Put your Gemini API Key"
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

def process_image(image):
    """Process image for Gemini API"""
    # Convert Gradio image (numpy array) to bytes
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

    # Convert to base64
    return base64.b64encode(image_bytes).decode('utf-8')

def recognize_asl(image):
    """Recognize ASL letter from image using Gemini"""
    if image is None:
        return "Please capture an image first"

    try:
        # Prepare image and prompt
        base64_image = process_image(image)
        prompt = """
        Identify the American Sign Language (ASL) letter shown in this image.
        Only respond with the single letter (A, B, C, D, or E) if recognized.
        If unclear, respond 'Unknown'.
        Do not include any other text or explanation.

        Image:
        """

        # Call Gemini API
        response = model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": base64_image}],
            stream=False
        )

        # Process response
        result = response.text.strip().upper()
        valid_letters = ['A', 'B', 'C', 'D', 'E']

        return result if result in valid_letters else "Unknown"

    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# 🖐️ ASL Letter Recognition")
    gr.Markdown("Show ASL signs for A, B, C, D, or E")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(sources=["webcam", "upload"],
                                  label="Capture ASL Sign",
                                  height=300)
            btn = gr.Button("Recognize Letter")

        with gr.Column():
            output = gr.Textbox(label="Recognized Letter",
                               placeholder="Letter will appear here")

    examples = gr.Examples(
        examples=["./A_test.jpg", "./B_test.jpg", "./C_test.jpg"],
        inputs=image_input,
        label="Example Images"
    )

    btn.click(recognize_asl, inputs=image_input, outputs=output)
    image_input.change(recognize_asl, inputs=image_input, outputs=output)

app.launch(debug=True)



