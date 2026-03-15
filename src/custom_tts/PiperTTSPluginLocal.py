import subprocess
import tempfile
import os
import numpy as np
import asyncio
import wave
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit import rtc

class PiperTTSPlugin(tts.TTS):
    def __init__(self, piper_path, model_path, speed=1.0, sample_rate=22050):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=sample_rate, num_channels=1)
        self.piper_path = piper_path
        self.model_path = model_path
        self.speed = speed
        self._sample_rate = sample_rate # 实测得piper（默认配置）的结果文件的采样率为22050 + 单声道
        self._num_channels = 1 

    def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return PiperStream(self, text, conn_options)

def normalize_text(text: str) -> str:
    return (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("-", "-")   # U+2011
            .replace("–", "-")   # en dash
            .replace("—", "-")   # em dash
    )

class PiperStream(tts.ChunkedStream):
    def __init__(self, plugin: "PiperTTSPlugin", text: str, conn_options):
        _input_text=normalize_text(text)
        super().__init__(tts=plugin, input_text=_input_text, conn_options=conn_options) 
        self.plugin = plugin
        self.text = _input_text

    # 注：原tts中chunkedStream的_run签名已经变了，增加了output_emitter)
    async def _run(self, output_emitter:tts.AudioEmitter) -> None:
        req_id = f"piper-{id(self)}"
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_file = f.name

            # Very important: tell the framework what kind of audio is coming
            output_emitter.initialize(
                
                request_id=req_id,
                sample_rate=self.plugin._sample_rate,
                num_channels=1,
                mime_type="audio/pcm"                     # or "audio/wav" — pcm is safer here
                # seg_id=seg_id # 如果是非streaming模式，默认会置空segment_id（参initialize方法）再启动start_segment
            )
            
            await asyncio.get_event_loop().run_in_executor(None, self._generate, temp_file)
            # 读取音频文件数据
            audio_data = await asyncio.get_event_loop().run_in_executor(None, self._read, temp_file)
            if not audio_data: 
                raise RuntimeError("Piper produced empty audio")
            #print(f"[Piper] audio_data length: {len(audio_data)} bytes")
            frame = rtc.AudioFrame(
                data=audio_data,
                sample_rate=self.plugin._sample_rate,
                num_channels=1,
                samples_per_channel=len(audio_data) // 2
            )
            output_emitter.push(audio_data)
        except Exception as e:
            print(f"Piper Error: {e}")
        finally:
            try:
                if os.path.exists(temp_file):                    
                    print(f"Temp WAV file size: {os.path.getsize(temp_file)} bytes")
                    os.unlink(temp_file)
            except:
                pass

    def _generate(self, output_file):
        # run a standalone sub process (piper exec). 将文本作为stdin，输出stdout/err（同时写到output_file中）
        subprocess.run([
            # if you are reading this, you can modify the command line arguments to fit your needs
            # https://github.com/OHF-Voice/piper1-gpl/tree/main/docs
            self.plugin.piper_path,
            "--model", self.plugin.model_path,
            "--length_scale", str(self.plugin.speed), # https://github.com/rhasspy/piper/discussions/199
            "--output_file", output_file
        ], input=self.input_text, text=True, check=True, 
        errors="ignore", # 
        encoding="utf-8" # Win下默认是GBK (cp936)，有些特殊字符（如’ (U+2019)，- (U+2011 non-breaking hyphen)，— (U+2014 em dash) stdio会无法编码）
        ) 

    def _read(self, wav_file):
        with wave.open(wav_file, 'rb') as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if w.getnchannels() == 2:
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
            return audio.tobytes()
