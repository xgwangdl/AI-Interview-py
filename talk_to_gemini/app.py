import asyncio
import base64
import json
import os
import pathlib
from typing import AsyncGenerator, Literal

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastrtc import (
    AsyncStreamHandler,
    Stream,
    get_twilio_turn_credentials,
    wait_for_item,
)
from google import genai
from google.genai import types
from google.genai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
)
from gradio.utils import get_space
from pydantic import BaseModel

current_dir = pathlib.Path(__file__).parent

load_dotenv()


def encode_audio(data: np.ndarray) -> str:
    """Encode Audio data to send to the server"""
    return base64.b64encode(data.tobytes()).decode("UTF-8")


class GeminiHandler(AsyncStreamHandler):
    """Handler for the Gemini API"""

    def __init__(
        self,
        expected_layout: Literal["mono"] = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: int = 480,
    ) -> None:
        super().__init__(
            expected_layout,
            output_sample_rate,
            output_frame_size,
            input_sample_rate=16000,
        )
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.quit: asyncio.Event = asyncio.Event()

    def copy(self) -> "GeminiHandler":
        return GeminiHandler(
            expected_layout="mono",
            output_sample_rate=self.output_sample_rate,
            output_frame_size=self.output_frame_size,
        )

    async def start_up(self):
        if not self.phone_mode:
            await self.wait_for_args()
            language, select_type, voice_name = self.latest_args[1:]
        else:
            language, select_type, voice_name = "英语","1","Puck"

        client = genai.Client(
            api_key= os.getenv("GEMINI_API_KEY"),
            http_options={"api_version": "v1alpha"},
        )

        if select_type == "1":
            prompt = f"""你是一位专业的{language}考官，正在进行3分钟的口语测试对话。请保持对话自然流畅，按照以下要求进行：
                                1. 开始时简单问候
                                2. 询问1-2个日常问题（如天气、爱好）
                                3. 提出1个情景问题（如旅行计划、工作场景）
                                4. 根据回答适当追问
                                5. 3分钟后自然结束对话"""
        else:
            prompt = f"""你是一位资深的小学{language}口语外教，正在进行3分钟的口语测试教学。请保持对话自然流畅，按照以下要求进行：
                                1. 开始时简单问候
                                2. 教小朋友简单会话
                                3. 适当用简单并且慢的语言问小朋友问题"""

        config = LiveConnectConfig(
            response_modalities=["AUDIO"],  # type: ignore
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(
                   text=prompt
                )],
                role="system"
            )
        )
        async with client.aio.live.connect(
            model="gemini-2.0-flash-exp", config=config
        ) as session:
            async for audio in session.start_stream(
                stream=self.stream(), mime_type="audio/pcm"
            ):
                if audio.data:
                    array = np.frombuffer(audio.data, dtype=np.int16)
                    self.output_queue.put_nowait((self.output_sample_rate, array))

    async def stream(self) -> AsyncGenerator[bytes, None]:
        while not self.quit.is_set():
            try:
                audio = await asyncio.wait_for(self.input_queue.get(), 0.1)
                yield audio
            except (asyncio.TimeoutError, TimeoutError):
                pass

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        _, array = frame
        array = array.squeeze()
        audio_message = encode_audio(array)
        self.input_queue.put_nowait(audio_message)

    async def emit(self) -> tuple[int, np.ndarray] | None:
        return await wait_for_item(self.output_queue)

    def shutdown(self) -> None:
        self.quit.set()


stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=GeminiHandler(),
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=180 if get_space() else None,
    additional_inputs=[
        gr.Textbox(
            label="API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY") if not get_space() else "",
        ),
        gr.Dropdown(
            label="Voice",
            choices=[
                "Puck",
                "Charon",
                "Kore",
                "Fenrir",
                "Aoede",
            ],
            value="Puck",
        ),
    ],
)


class InputData(BaseModel):
    webrtc_id: str
    language: str
    select_type: str
    voice_name: str



app = FastAPI()

stream.mount(app)


@app.post("/input_hook")
async def _(body: InputData):
    stream.set_input(body.webrtc_id, body.language, body.select_type, body.voice_name)
    return {"status": "ok"}


@app.get("/gemini-voice")
async def index():
    rtc_config = get_twilio_turn_credentials() if get_space() else None
    html_content = (current_dir / "index.html").read_text(encoding='utf-8')
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import os

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)
