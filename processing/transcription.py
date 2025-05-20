import whisper

class SpeechToText:
    def __init__(self, audio_file, device):
        self.audio_file = audio_file
        self.device = device
        self.model = whisper.load_model("base").to(self.device)

    def transcribe(self):
        result = self.model.transcribe(self.audio_file, language="en", verbose=False)
        return result["segments"]