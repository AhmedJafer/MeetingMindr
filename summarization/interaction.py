from openai import OpenAI
from processing.diarization import SpeakerDiarization
from processing.transcription import SpeechToText
from processing.mapping import SpeakerTextMapper
from processing.formatting import TranscriptFormatter
from summarization.evaluator import SummaryEvaluator
import os
from moviepy import VideoFileClip
import markdown
from weasyprint import HTML

class CallInteraction:
    def __init__(self, audio_or_video_file, device, hugging_face_token,
                 api_key, base_url, model_name,
                 prompt_template=None,
                 evaluation_api_key=None, evaluation_base_url=None, evaluation_model_name=None,output_dir="Meeting_Summaries",
                 audio_output_dir="Extracted_Audio"):

        self.original_file = audio_or_video_file
        self.device = device
        self.hugging_face_token = hugging_face_token
        self.model_name = model_name
        self.output_dir = output_dir
        self.audio_output_dir = audio_output_dir
        os.makedirs(self.output_dir, exist_ok=True)


        self.Summary_model = OpenAI(api_key=api_key, base_url=base_url)
        self.system_prompt = prompt_template or self._default_prompt()

        self.evaluator = SummaryEvaluator(
            api_key=evaluation_api_key or api_key,
            base_url=evaluation_base_url or base_url,
            model_name=evaluation_model_name or model_name
        )


        if self.original_file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
            os.makedirs(self.audio_output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(self.original_file))[0]
            extracted_audio_path = os.path.join(self.audio_output_dir, f"{base_name}.wav")
            self.audio_file = self.extract_audio_from_video(self.original_file, extracted_audio_path)
        else:
            self.audio_file = self.original_file

    def extract_audio_from_video(self, video_path, output_audio_path):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path, codec='pcm_s16le')
        return output_audio_path

    def _default_prompt(self):
        return ("""You are an expert meeting assistant. Your task is to read the transcript of a meeting and generate a clear, concise, and structured summary\n\n.

            ## Required Sections in Your Summary\n
            1. Meeting Title and Date (if available)\n
            2. Attendees (with roles if mentioned)\n
            3. Meeting Duration (if available)\n
            4. Executive Summary (2-3 sentences capturing the most important takeaway)\n
            5. Main Topics Discussed (prioritized by importance)\n
            6. Key Decisions Made (clearly marked as decisions)\n
            7. Action Items (with owners, deadlines, and priority levels if stated)\n
            8. Open Questions or Follow-Ups (items requiring further discussion)\n
            9. Next Steps (immediate actions and upcoming meetings)\n\n
            
            ## Formatting Guidelines\n
            * Use hierarchical structure with clear headings\n
            * Highlight critical information in **bold**\n
            * Use bullet points for lists and action items\n
            * Present action items in a bullets point format \n
            * Indicate priority levels for action items (High/Medium/Low) when context suggests importance\n\n
            
            ## Content Guidelines\n
            * Focus on substance: omit small talk, technical difficulties, and off-topic discussions\n
            * Preserve the key insights from discussions, not just final decisions\n
            * When technical terms or acronyms are used, briefly explain them in parentheses if context is provided\n
            * When discussions are ambiguous, note this rather than making assumptions\n
            * If multiple perspectives were shared on an important topic, briefly note the different viewpoints\n
            * For lengthy meetings (>30 minutes of transcript), adjust detail level to keep summary under 500 words\n
            * For shorter meetings, aim for comprehensive coverage while staying under 250 words\n
            
            ## Context Awareness\n\n
            * Identify recurring topics from previous meetings if mentioned\n
            * Note any references to external documents, projects, or deadlines\n
            * Capture emotional tone of important discussions only when clearly relevant (e.g., "team expressed strong concerns about timeline")\n
            
            
            """)

    def _update_prompt_on_failure(self, summary, feedback):
        updated_prompt = self.system_prompt + f"\n\n## Previous summary was rejected\nYou just tried to reply, but the quality control rejected your summary\n"
        updated_prompt += f"## Your attempted summary:\n{summary}\n\n"
        updated_prompt += f"## Reason for rejection:\n{feedback}\n\n"
        return updated_prompt

    def _generate_summary(self, transcript):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": transcript}
        ]
        response = self.Summary_model.chat.completions.create(
            messages=messages,
            model=self.model_name
        )
        return response.choices[0].message.content

    def _save_to_pdf(self, markdown_content, title, output_path):
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content)

        # Wrap HTML with title and simple style
        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 30px; }}
                h1 {{ font-size: 28px; }}
                h2 {{ font-size: 24px; }}
                h3 {{ font-size: 20px; }}
                ul {{ margin-left: 20px; }}
                li {{ margin-bottom: 5px; }}
                strong {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            {html_content}
        </body>
        </html>
        """

        # Generate PDF with weasyprint
        HTML(string=html).write_pdf(output_path)

    def _evaluate_and_improve_summary(self, transcript, initial_summary, max_retries=5):
        summary = initial_summary
        for attempt in range(max_retries):
            evaluation = self.evaluator.evaluate(transcript, summary)
            if evaluation.is_acceptable:
                return summary, evaluation.feedback

            updated_prompt = self._update_prompt_on_failure(transcript, summary, evaluation.feedback)
            retry_messages = [
                {"role": "system", "content": updated_prompt},
                {"role": "user", "content": transcript}
            ]
            summary = self.Summary_model.chat.completions.create(
                messages=retry_messages, model=self.model_name
            ).choices[0].message.content

        return summary, evaluation.feedback

    def summarize_call(self):
        diarizer = SpeakerDiarization(self.audio_file, self.device, self.hugging_face_token)
        transcriber = SpeechToText(self.audio_file, self.device)

        segments = transcriber.transcribe()
        speaker_segments = diarizer.diarize()

        mapper = SpeakerTextMapper(speaker_segments, segments)
        mapped = mapper.map_speakers()

        formatted_transcript = TranscriptFormatter(mapped).format_transcript()

        initial_summary = self._generate_summary(formatted_transcript)
        final_summary, feedback = self._evaluate_and_improve_summary(formatted_transcript, initial_summary)

        base_name = os.path.splitext(os.path.basename(self.audio_file))[0]
        summary_path = os.path.join(self.output_dir, f"{base_name}_summary.pdf")
        transcript_path = os.path.join(self.output_dir, f"{base_name}_transcript.pdf")

        self._save_to_pdf(final_summary.strip(), "Meeting Summary", summary_path)
        self._save_to_pdf(formatted_transcript.strip(), "Full Meeting Transcript", transcript_path)

        return formatted_transcript, final_summary, feedback
