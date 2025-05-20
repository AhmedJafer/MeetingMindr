import os
import gradio as gr
import torch
from summarization.interaction import CallInteraction
from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model_name = "gemini-2.0-flash"
hugging_face_token = os.getenv("hugging_face_token")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_uploaded_audio(audio_file):
    interaction = CallInteraction(
        audio_file=audio_file,
        device=device,
        hugging_face_token=hugging_face_token,
        api_key=google_api_key,
        base_url=base_url,
        model_name=model_name
    )
    return interaction.summarize_call()

with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Call Summarizer: Record or Upload Audio")
    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Upload Call")
    transcript_output = gr.Textbox(label="Transcript", lines=8)
    summary_output = gr.Textbox(label="Summary", lines=8)
    feedback_output = gr.Textbox(label="LLM Evaluation Feedback", lines=5)
    summarize_button = gr.Button("Summarize")

    summarize_button.click(
        fn=process_uploaded_audio,
        inputs=audio_input,
        outputs=[transcript_output, summary_output, feedback_output]
    )

demo.launch()
