

class TranscriptFormatter:
    def __init__(self, final_output):
        self.final_output = final_output

    def format_transcript(self):
        # Map for speaker names
        speaker_map = {}
        speaker_counter = 1

        # Prepare for grouped speaker text
        grouped_transcript = []

        current_speaker = None
        accumulated_text = []

        for entry in self.final_output:
            speaker_raw = entry["speaker"]

            # Map speakers
            if speaker_raw not in speaker_map:
                speaker_map[speaker_raw] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            speaker_label = speaker_map[speaker_raw]

            # If speaker changes, save previous accumulated text
            if speaker_label != current_speaker:
                if current_speaker is not None:  # Only save if it's not the first entry
                    grouped_transcript.append(f"{current_speaker}: {' '.join(accumulated_text)}")
                current_speaker = speaker_label
                accumulated_text = [entry['text']]  # Start accumulating new speaker's text
            else:
                accumulated_text.append(entry['text'])

        # Don't forget to append the last speaker's text
        if accumulated_text:
            grouped_transcript.append(f"{current_speaker}: {' '.join(accumulated_text)}")

        # Join everything into the final output format
        return "\n\n".join(grouped_transcript)