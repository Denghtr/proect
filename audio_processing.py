import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
from scipy.fft import fft, ifft

class AudioProcessor:
    def load_audio(self, file_path):
        sample_rate, data = wav.read(file_path)
        return sample_rate, data

    def save_audio(self, file_path, sample_rate, cleaned_data):
        cleaned_data = np.real(cleaned_data).astype(np.int16)
        wav.write(file_path, sample_rate, cleaned_data)

    def apply_window_function(self, data):
        window = np.hamming(len(data))
        return data * window

    def clean_and_amplify_audio(self, input_file, output_file, low_cutoff=85, high_cutoff=3000, gain=1):
        sample_rate, data = self.load_audio(input_file)

        # Применение шумоподавления
        reduced_noise = nr.reduce_noise(y=data.astype(float), sr=sample_rate, 
                                        prop_decrease=0.9, stationary=False)

        # Применение оконной функции
        windowed_data = self.apply_window_function(reduced_noise)

        # Применение дискретного преобразования Фурье
        fft_data = fft(windowed_data)

        # Создание маски для частот
        freq_bins = np.fft.fftfreq(len(fft_data), d=1/sample_rate)
        mask = (np.abs(freq_bins) >= low_cutoff) & (np.abs(freq_bins) <= high_cutoff)

        # Увеличение амплитуды частот в диапазоне 85-3000 Гц
        fft_data[mask] *= gain

        # Применение обратного преобразования Фурье
        cleaned_data = ifft(fft_data)

        self.save_audio(output_file, sample_rate, cleaned_data)