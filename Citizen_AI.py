import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer efficiently
model_name = "ibm-granite/granite-3.2-2b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Core generation function
def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

# Use-case-specific prompt generators
def city_analysis(city_name):
    prompt = f"Provide a detailed analysis of {city_name} including:\n1. Crime Index and safety statistics\n2. Accident rates and traffic safety information\n3. Overall safety assessment\n\nCity: {city_name}\nAnalysis:"
    return generate_response(prompt, max_length=1000)

def citizen_interaction(query):
    prompt = f"As a government assistant, provide accurate and helpful information about the following citizen query related to public services, government policies, or civic issues:\n\nQuery: {query}\nResponse:"
    return generate_response(prompt, max_length=1000)

# Custom CSS for gradient background and UI tweaks
custom_css = """
/* Full page gradient background */
html, body, #root, .gradio-container {
    height: 100%;
    background: linear-gradient(to right, #ff69b4, #8a2be2) !important;
    color: white;
    margin: 0;
    padding: 0;
}

/* General UI tweaks */
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3, h4, label {
    color: white !important;
}

/* Inputs and Buttons */
textarea, input, .gr-button {
    border-radius: 10px !important;
    border: none !important;
    color: #333;
}

.gr-button {
    background: #ff69b4 !important;
    color: white !important;
    font-weight: bold;
}

.gr-button:hover {
    background: #e754b5 !important;
}
"""

# Build the interface
with gr.Blocks(css=custom_css) as app:
    gr.Markdown("## 🌆 City Analysis & 🏛️ Citizen Services AI", elem_id="title")

    with gr.Tabs():
        with gr.TabItem("City Analysis"):
            with gr.Row():
                with gr.Column():
                    city_input = gr.Textbox(
                        label="Enter City Name",
                        placeholder="e.g., New York, London, Mumbai...",
                        lines=1
                    )
                    analyze_btn = gr.Button("🔍 Analyze City")

                with gr.Column():
                    city_output = gr.Textbox(label="📊 City Analysis (Crime Index & Accidents)", lines=15)

            analyze_btn.click(city_analysis, inputs=city_input, outputs=city_output)

        with gr.TabItem("Citizen Services"):
            with gr.Row():
                with gr.Column():
                    citizen_query = gr.Textbox(
                        label="Your Query",
                        placeholder="Ask about public services, government policies, civic issues...",
                        lines=4
                    )
                    query_btn = gr.Button("ℹ️ Get Information")

                with gr.Column():
                    citizen_output = gr.Textbox(label="📬 Government Response", lines=15)

            query_btn.click(citizen_interaction, inputs=citizen_query, outputs=citizen_output)

# Launch app
app.launch(share=True)
