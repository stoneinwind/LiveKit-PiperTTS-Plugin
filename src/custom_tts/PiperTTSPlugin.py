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
            # 用了piper-tts的python库，但依然是将当前输入的text（可能是多个段落）全部处理完才返回 -- 非真实流
            chunks = await loop.run_in_executor(None, self._synthesize_chunks, config)
            if not chunks: 
                raise RuntimeError("Piper-tts produced empty audio chunks")
            for chunk in chunks:                
                output_emitter.push(chunk)                 
        except Exception as e:
            print(f"Piper execeptions: {e}")
            _silence_data = np.zeros(22050, dtype=np.int16).tobytes()
            output_emitter.push(_silence_data)

    # this is not a real streaming implementation, so if you are reading this, check https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md and PiperVoice.synthesize
    def _synthesize_chunks(self, config):
        chunks = []
        chunk_count = 0
        total_bytes  = 0
        for chunk in self.plugin._voice.synthesize(self.input_text, syn_config=config):
            audio_bytes = chunk.audio_int16_bytes
            if chunk.sample_channels == 2:
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
                audio_bytes = audio.tobytes()
            # Debug only..
            if len(audio_bytes) == 0:
                print(f"Empty chunk for phonemes: {chunk.phonemes}")
                continue
                
            chunks.append(audio_bytes)
            chunk_count += 1
            total_bytes += len(audio_bytes)
            print(f"Chunk {chunk_count}: {len(audio_bytes)} bytes, phonemes len={len(chunk.phonemes)}")

        print(f"Total: {chunk_count} chunks, {total_bytes} bytes")
        if total_bytes == 0:
            print("WARNING: No audio generated at all!")            
        return chunks
