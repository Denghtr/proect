import json
import difflib
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
from scipy.fft import fft, ifft
import os
import vosk

# Функция для загрузки команд из JSON-файла
def load_commands_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Функция для поиска наиболее близкого совпадения
def find_best_match(input_text: str, labels) -> tuple:
    best_match = None
    best_ratio = 0.0
    
    for label in labels:
        match_ratio = difflib.SequenceMatcher(None, input_text, label).ratio()
        if match_ratio > best_ratio:
            best_ratio = match_ratio
            best_match = label

    return best_match, best_ratio

# Функции для обработки аудио
def load_audio(file_path):
    sample_rate, data = wav.read(file_path)
    return sample_rate, data

def save_audio(file_path, sample_rate, cleaned_data):
    cleaned_data = np.real(cleaned_data).astype(np.int16)
    wav.write(file_path, sample_rate, cleaned_data)

def apply_window_function(data):
    window = np.hamming(len(data))
    return data * window

def clean_and_amplify_audio(input_file, output_file, low_cutoff=85, high_cutoff=3000, gain=1):
    sample_rate, data = load_audio(input_file)

    # Применение шумоподавления
    reduced_noise = nr.reduce_noise(y=data.astype(float), sr=sample_rate, 
                                     prop_decrease=0.9, stationary=False)

    # Применение оконной функции
    windowed_data = apply_window_function(reduced_noise)

    # Применение дискретного преобразования Фурье
    fft_data = fft(windowed_data)

    # Создание маски для частот
    freq_bins = np.fft.fftfreq(len(fft_data), d=1/sample_rate)
    mask = (np.abs(freq_bins) >= low_cutoff) & (np.abs(freq_bins) <= high_cutoff)

    # Увеличение амплитуды частот в диапазоне 85-3000 Гц
    fft_data[mask] *= gain

    # Применение обратного преобразования Фурье
    cleaned_data = ifft(fft_data)

    save_audio(output_file, sample_rate, cleaned_data)

# Функция для распознавания текста из аудио с использованием Vosk
def transcribe_audio(file_path, model_path, commands):
    if not os.path.exists(model_path):
        return f"Model path not found: {model_path}"

    model = vosk.Model(model_path)
    rec = vosk.KaldiRecognizer(model, 16000)

    with open(file_path, "rb") as f:
        wf = f.read()
        rec.AcceptWaveform(wf)  
        result = rec.Result()  

        result_dict = json.loads(result)
        recognized_text = result_dict.get('text', '')  
        
        best_match, match_ratio = find_best_match(recognized_text, commands)

        if best_match:
            return f"Recognized: {recognized_text}, Command: {best_match}, Match Ratio: {match_ratio:.2f}"
        else:
            return f"Recognized: {recognized_text}, No matches found."

# Функция для обработки всех файлов в папке
def process_folder(folder_path, model_path, commands, log_file):
    log_data = []
    
    # Итерируем по всем файлам в папке
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):  # Обрабатываем только wav файлы
            input_file = os.path.join(folder_path, file_name)
            output_file = os.path.join(folder_path, f"cleaned_{file_name}")
            
            # Очистка и усиление аудио
            clean_and_amplify_audio(input_file, output_file, low_cutoff=85, high_cutoff=3000, gain=1)
            
            # Распознавание аудио
            result = transcribe_audio(output_file, model_path, commands)
            
            # Добавляем информацию в лог
            log_data.append({
                "file": file_name,
                "result": result
            })
            print(f"Processed {file_name}: {result}")
    
    # Сохраняем лог в файл
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)

# Пример использования
commands_json_file = 'commands.json'
commands = load_commands_from_json(commands_json_file)

folder_path = 'audio_files'  # Папка с аудиофайлами
model_path = "model_small"  # Путь к модели Vosk
log_file = 'log.json'  # Файл лога

process_folder(folder_path, model_path, list(commands.keys()), log_file)
