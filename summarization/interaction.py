from openai import OpenAI
from processing.diarization import SpeakerDiarization
from processing.transcription import SpeechToText
from processing.mapping import SpeakerTextMapper
from processing.formatting import TranscriptFormatter
from summarization.evaluator import SummaryEvaluator


class CallInteraction:
    def __init__(self, audio_file, device, hugging_face_token,
                 api_key, base_url, model_name,
                 prompt_template=None,
                 evaluation_api_key=None, evaluation_base_url=None, evaluation_model_name=None):
        self.audio_file = audio_file
        self.device = device
        self.hugging_face_token = hugging_face_token
        self.model_name = model_name
        self.Summary_model = OpenAI(api_key=api_key, base_url=base_url)
        self.system_prompt = prompt_template or self.default_prompt()
        self.evaluator = SummaryEvaluator(
            api_key=evaluation_api_key or api_key,
            base_url=evaluation_base_url or base_url,
            model_name=evaluation_model_name or model_name
        )

    def default_prompt(self):
        return (
            "You are a summarization model. Your task is to analyze the following conversation "
            "or call transcript and generate a concise summary that captures the most important takeaways, "
            "key decisions, action items, and any notable concerns or questions raised during the discussion.\n\n"
            "Guidelines:\n"
            "- Focus only on relevant and impactful information.\n"
            "- Do not include small talk or greetings.\n"
            "- Clearly identify who made key points or decisions (if possible).\n"
            "- Present the summary in bullet points or short paragraphs for clarity.\n"
            "- Maintain a professional and objective tone.\n\n"
            "Expected Output:\n"
            "- Summary of the most important points\n"
            "- Action items (if any)\n"
            "- Key decisions made\n"
            "- Any questions or concerns raised\n\n"
            "Transcript:\n\n"
        )

    def updated_system_prompt(self, transcript, summary, feedback):
        updated_prompt = self.system_prompt + f"\n\n## Previous summary was rejected\nYou just tried to reply, but the quality control rejected your summary\n"
        updated_prompt += f"## Your attempted summary:\n{summary}\n\n"
        updated_prompt += f"## Reason for rejection:\n{feedback}\n\n"
        return updated_prompt

    def summarize_call(self):
        diarizer = SpeakerDiarization(self.audio_file, self.device, self.hugging_face_token)
        transcriber = SpeechToText(self.audio_file, self.device)
        segments = transcriber.transcribe()
        speaker_segments = diarizer.diarize()

        mapper = SpeakerTextMapper(speaker_segments, segments)
        mapped = mapper.map_speakers()
        formatted = TranscriptFormatter(mapped).format_transcript()

        with open("transcription.txt", "w", encoding="utf-8") as f:
            f.write(formatted)

        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": formatted}]
        summary = self.Summary_model.chat.completions.create(messages=messages, model=self.model_name).choices[0].message.content
        evaluation = self.evaluator.evaluate(formatted, summary)

        if not evaluation.is_acceptable:
            retry_messages = [{"role": "system", "content": self.updated_system_prompt(formatted, summary, evaluation.feedback)}, {"role": "user", "content": formatted}]
            summary = self.Summary_model.chat.completions.create(messages=retry_messages, model=self.model_name).choices[0].message.content
            evaluation = self.evaluator.evaluate(formatted, summary)

        return formatted, summary, evaluation.feedback
