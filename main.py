import argparse
import os
from dotenv import load_dotenv
from summarization.interaction import CallInteraction
import torch
import warnings
warnings.filterwarnings("ignore")

def main():

    load_dotenv()

    parser = argparse.ArgumentParser(description="Summarize a meeting from an audio or video file")
    parser.add_argument("--audio_or_video_file", required=True, help="Path to the audio or video file")
    parser.add_argument("--device", help="Device for processing (e.g. cpu or cuda). Defaults to best available.")
    parser.add_argument("--hugging_face_token", help="Your Hugging Face token")
    parser.add_argument("--api_key", help="LLM API key")
    parser.add_argument("--base_url", default="https://generativelanguage.googleapis.com/v1beta/openai/", help="LLM base URL")
    parser.add_argument("--model_name", required=True, help="LLM model name (e.g. gpt-4)")
    parser.add_argument("--output_dir", default="Meeting_Summaries", help="Directory to save the final output")
    parser.add_argument("--audio_output_dir", default="Extracted_Audio", help="Directory to save extracted audio")
    parser.add_argument("--evaluation_api_key", help="Optional evaluation API key")
    parser.add_argument("--evaluation_base_url", help="Optional evaluation base URL")
    parser.add_argument("--evaluation_model_name", help="Optional evaluation model name")

    args = parser.parse_args()


    hugging_face_token = args.hugging_face_token or os.getenv("HUGGING_FACE_TOKEN")
    api_key = args.api_key or os.getenv("LLM_API_KEY")

    if not hugging_face_token:
        raise ValueError("Hugging Face token not provided. Use --hugging_face_token or set HUGGING_FACE_TOKEN in .env")

    if not api_key:
        raise ValueError("OpenAI API key not provided. Use --api_key or set OPENAI_API_KEY in .env")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    call = CallInteraction(
        audio_or_video_file=args.audio_or_video_file,
        device=device,
        hugging_face_token=hugging_face_token,
        api_key=api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        output_dir=args.output_dir,
        audio_output_dir=args.audio_output_dir,
        evaluation_api_key=args.evaluation_api_key or os.getenv("EVALUATION_API_KEY"),
        evaluation_base_url=args.evaluation_base_url or os.getenv("EVALUATION_BASE_URL"),
        evaluation_model_name=args.evaluation_model_name or os.getenv("EVALUATION_MODEL_NAME")
    )

    transcript, summary, feedback = call.summarize_call()
    print("Summary and transcript generated successfully.")


if __name__ == "__main__":
    main()
