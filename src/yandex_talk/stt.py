import os
import queue
import json
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from pydub.silence import split_on_silence
from webrtcvad import Vad
import threading
import time
import wave
import tempfile
from collections import defaultdict

from src.yandex_talk.constants import MODEL_PATH


class SpeechRecognizer:
    def __init__(self, model_path="vosk-model-small-en-us-0.15", sample_rate=16000):
        """
        Инициализация распознавателя речи

        :param model_path: путь к модели Vosk
        :param sample_rate: частота дискретизации аудио
        """
        if not os.path.exists(model_path):
            print(f"Модель {model_path} не найдена. Пожалуйста, скачайте модель с https://alphacephei.com/vosk/models")
            exit(1)

        self.model = Model(model_path)
        self.sample_rate = sample_rate
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.vad = Vad(3)  # Агрессивный режим VAD
        self.speaker_segments = defaultdict(list)
        self.current_speaker = 0
        self.last_speaker_change = time.time()
        self.speaker_silence_threshold = 1.5  # секунды молчания для смены спикера

    def _process_audio_chunk(self, audio_data):
        """
        Обработка аудио фрагмента и распознавание речи

        :param audio_data: numpy массив с аудио данными
        """
        if self.recognizer.AcceptWaveform(audio_data.tobytes()):
            result = json.loads(self.recognizer.Result())
            if 'text' in result and result['text']:
                # Проверяем, нужно ли сменить спикера
                if time.time() - self.last_speaker_change > self.speaker_silence_threshold:
                    self.current_speaker = (self.current_speaker + 1) % 2  # Просто переключаем между 0 и 1
                    self.last_speaker_change = time.time()

                # Сохраняем текст с идентификатором спикера
                self.speaker_segments[self.current_speaker].append({
                    'text': result['text'],
                    'timestamp': time.time()
                })
                print(f"Speaker {self.current_speaker}: {result['text']}")

    def _audio_callback(self, indata, frames, time, status):
        """
        Callback функция для записи аудио с микрофона
        """
        if status:
            print(status, flush=True)
        if self.is_listening:
            self.audio_queue.put(indata.copy())

    def _process_audio_stream(self):
        """
        Обработка аудио потока в отдельном потоке
        """
        while self.is_listening or not self.audio_queue.empty():
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)

                # Конвертируем в 16-bit PCM
                audio_chunk = (audio_chunk * 32767).astype(np.int16)

                # Применяем VAD для обнаружения речи
                if self._has_speech(audio_chunk):
                    self._process_audio_chunk(audio_chunk)
            except queue.Empty:
                continue

    def _has_speech(self, audio_chunk):
        """Проверка наличия речи с помощью WebRTC VAD"""
        # 1. Конвертируем в 16-bit PCM моно, если это ещё не сделано
        if audio_chunk.dtype != np.int16:
            audio_chunk = (audio_chunk * 32767).astype(np.int16)

        # 2. Проверяем частоту дискретизации (должна быть 8000, 16000, 32000 или 48000)
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}. Must be 8000, 16000, 32000 or 48000")

        # 3. Убедимся, что длина фрейма соответствует 10, 20 или 30 мс
        frame_duration_ms = 30  # Выбираем 30 мс для надёжности
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)

        # Если аудио короче нужного фрейма - дополняем нулями
        if len(audio_chunk) < frame_size:
            padded = np.zeros(frame_size, dtype=np.int16)
            padded[:len(audio_chunk)] = audio_chunk
            audio_chunk = padded
        elif len(audio_chunk) > frame_size:
            audio_chunk = audio_chunk[:frame_size]

        # 4. Конвертируем в bytes и проверяем речь
        audio_bytes = audio_chunk.tobytes()
        return self.vad.is_speech(audio_bytes, self.sample_rate)

    def start_listening(self):
        """
        Начать запись и распознавание речи
        """
        self.is_listening = True
        self.audio_queue = queue.Queue()

        # Запускаем поток для обработки аудио
        processing_thread = threading.Thread(target=self._process_audio_stream)
        processing_thread.daemon = True
        processing_thread.start()

        # Начинаем запись с микрофона
        with sd.InputStream(samplerate=self.sample_rate,
                            channels=1,
                            dtype='float32',
                            blocksize=4096,
                            callback=self._audio_callback):
            print("Listening... Press Ctrl+C to stop.")
            while self.is_listening:
                sd.sleep(100)

    def stop_listening(self):
        """
        Остановить запись и распознавание речи
        """
        self.is_listening = False
        print("Stopped listening.")

    def get_transcript(self):
        """
        Получить полную расшифровку с разделением по спикерам
        """
        return dict(self.speaker_segments)

    def save_audio_to_file(self, audio_data, filename):
        """
        Сохранить аудио данные в WAV файл
        """
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())


def main():
    # Инициализация распознавателя
    recognizer = SpeechRecognizer(model_path=str(MODEL_PATH))

    try:
        # Запуск прослушивания в отдельном потоке
        listen_thread = threading.Thread(target=recognizer.start_listening)
        listen_thread.daemon = True
        listen_thread.start()

        # Главный поток ждет завершения
        while listen_thread.is_alive():
            listen_thread.join(0.1)
    except KeyboardInterrupt:
        recognizer.stop_listening()

    # Вывод результатов
    print("\nFinal transcript:")
    for speaker, segments in recognizer.get_transcript().items():
        print(f"\nSpeaker {speaker}:")
        for segment in segments:
            print(f"[{segment['timestamp']}] {segment['text']}")


if __name__ == "__main__":
    main()
