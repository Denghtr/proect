import json
import difflib
import os
import vosk

class CommandRecognizer:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        self.model = vosk.Model(model_path)
    
    def load_commands_from_json(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def find_best_match(self, input_text: str, labels) -> tuple:
        best_match = None
        best_ratio = 0.0
        for label in labels:
            match_ratio = difflib.SequenceMatcher(None, input_text, label).ratio()
            if match_ratio > best_ratio:
                best_ratio = match_ratio
                best_match = label
        return best_match, best_ratio

    def transcribe_audio(self, file_path, commands):
        rec = vosk.KaldiRecognizer(self.model, 16000)
        with open(file_path, "rb") as f:
            wf = f.read()
            rec.AcceptWaveform(wf)
            result = rec.Result()

            result_dict = json.loads(result)
            recognized_text = result_dict.get('text', '')

            best_match, match_ratio = self.find_best_match(recognized_text, commands)
            if best_match:
                return f"Recognized: {recognized_text}, Command: {best_match}, Match Ratio: {match_ratio:.2f}"
            else:
                return f"Recognized: {recognized_text}, No matches found."
