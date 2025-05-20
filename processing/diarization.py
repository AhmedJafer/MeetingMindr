import torch
from pyannote.audio import Pipeline
from functools import partial

class SpeakerDiarization:
    def __init__(self, audio_file, device, hugging_face_token):
        self.audio_file = audio_file
        self.device = device
        self.hugging_face_token = hugging_face_token
        self.pipeline = self._load_diarization_pipeline()

    def _load_diarization_pipeline(self):
        # Patch for pyannote
        original_load = torch.load
        torch.load = partial(original_load, weights_only=False)

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hugging_face_token
        ).to(self.device)

        # Restore original torch.load
        torch.load = original_load

        return pipeline

    def diarize(self):
        diarization = self.pipeline(self.audio_file)
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return speaker_segments