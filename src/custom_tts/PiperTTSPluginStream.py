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
        # according to the docs if you enable cuda you need onnxruntime-gpu package, read the docs
        # if no GPU onnx version: onnxruntime\capi\onnxruntime_inference_collection.py:
        #   Specified provider 'CUDAExecutionProvider' is not in available provider names.
        #   Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
        # with onnx GPU runtime installed,:
        #   ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
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
                mime_type="audio/pcm"                     # or "audio/wav" — pcm is safer here
            )

            config = SynthesisConfig(
                volume=self.plugin.volume,
                length_scale=self.plugin.speed,
                noise_scale=self.plugin.noise_scale,
                noise_w_scale=self.plugin.noise_w,
                normalize_audio=True # True will depress abnormal volumes
            )            
            loop = asyncio.get_running_loop() # instead of event loop

            import queue
            q = queue.Queue(maxsize=10) # backpressure - if maxsize reached, block producer
            STOP = object()
            def producer():
                try:
                    for chunk in self.plugin._voice.synthesize(
                        self.input_text,
                        syn_config=config
                        ):
                        q.put(chunk.audio_int16_bytes)
                finally:
                    q.put(STOP)
            loop.run_in_executor(None, producer)
            while True:
                chunk_bytes = await loop.run_in_executor(None, q.get)
                if chunk_bytes is STOP:
                    break
                output_emitter.push(chunk_bytes)            
        except Exception as e:
            print(f"Piper execeptions: {e}")
            _silence_data = np.zeros(22050, dtype=np.int16).tobytes()
            output_emitter.push(_silence_data)
