from datetime import timedelta

class SpeakerTextMapper:
    def __init__(self, speaker_segments, transcribed_segments):
        self.speaker_segments = speaker_segments
        self.transcribed_segments = transcribed_segments

    def find_speaker(self, start_time, end_time):
        for segment in self.speaker_segments:
            # If the midpoint of the whisper segment lies inside a diarization segment
            mid_point = (start_time + end_time) / 2
            if segment["start"] <= mid_point <= segment["end"]:
                return segment["speaker"]
        return "UNKNOWN"

    def map_speakers(self):
        final_output = []
        for seg in self.transcribed_segments:
            speaker = self.find_speaker(seg['start'], seg['end'])
            text = seg['text'].strip()
            start_time = str(timedelta(seconds=int(seg['start'])))
            end_time = str(timedelta(seconds=int(seg['end'])))
            final_output.append({
                "speaker": speaker,
                "start": start_time,
                "end": end_time,
                "text": text
            })
        return final_output