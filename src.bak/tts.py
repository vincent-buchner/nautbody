from chatterbox.tts_turbo import ChatterboxTurboTTS


class TTSService:
    def __init__(self, audio_ref_path: str) -> None:
        self._model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        self._audio_ref_path = audio_ref_path
        # Prepare the voice-clone conditionals from the reference audio once up
        # front. If we instead pass audio_prompt_path on every generate() call,
        # the model re-derives these conditionals (reloading + re-embedding the
        # reference wav) on every single turn, adding several seconds of extra
        # "loading" after each response.
        self._model.prepare_conditionals(audio_ref_path)

    def generate(self, text: str):
        wav = self._model.generate(text=text)
        return wav

    @property
    def sample_rate(self):
        return self._model.sr
