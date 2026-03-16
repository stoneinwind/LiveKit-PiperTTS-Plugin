import numpy as np
import asyncio
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit import rtc
from piper import PiperVoice, SynthesisConfig

def normalize_text(text: str) -> str:
    return (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("-", "-")   # U+2011
            .replace("–", "-")   # en dash
            .replace("—", "-")   # em dash
    )

class PiperTTSPlugin(tts.TTS):
    def __init__(self, model, speed=1.0, volume=1.0, noise_scale=0.667, noise_w=0.8, use_cuda=False):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=22050, num_channels=1)
        self.model_path = model
        self.speed = speed
        self.volume = volume
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.use_cuda = use_cuda
        self._voice = None
        self._load_voice()

    def _load_voice(self):
        self._voice = PiperVoice.load(self.model_path, use_cuda=self.use_cuda)
        
    def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        _input_text=normalize_text(text)
        return PiperApiStream(self, _input_text, conn_options)

class PiperApiStream(tts.ChunkedStream):
    def __init__(self, plugin, text, conn_options):
        super().__init__(tts=plugin, input_text=text, conn_options=conn_options)
        self.plugin = plugin

    async def _run(self, output_emitter:tts.AudioEmitter):
        try:     
            req_id = f"pipertts-{id(self)}"
            output_emitter.initialize(                
                request_id=req_id,
                sample_rate=self.plugin._sample_rate,
                num_channels=1,
                mime_type="audio/pcm"
            )

            config = SynthesisConfig(
                volume=self.plugin.volume,
                length_scale=self.plugin.speed,
                noise_scale=self.plugin.noise_scale,
                noise_w_scale=self.plugin.noise_w,
                normalize_audio=True # True will depress abnormal volumes
            )            
            loop = asyncio.get_event_loop()

            # use PiperVoice's synthesize func (Synthesize one audio chunk per sentence from text)
            def synth_iter():
                for chunk in self.plugin._voice.synthesize(
                    self.input_text,
                    syn_config=config
                ):
                    yield chunk.audio_int16_bytes
            def safe_next(it):
                try:
                    return next(it)
                except StopIteration:
                    # 当迭代结束时，返回一个特殊标记（例如 None）
                    return None

            iterator = synth_iter()
            while True:
                try:
                    chunk_bytes = await loop.run_in_executor(None, safe_next, iterator)
                    if chunk_bytes is None:
                        break
                    output_emitter.push(chunk_bytes) 
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Synthesis error: {e}")
                    break            
        except Exception as e:
            print(f"Piper execeptions: {e}")
            _silence_data = np.zeros(22050, dtype=np.int16).tobytes()
            output_emitter.push(_silence_data)
