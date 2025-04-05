import asyncio
import base64
import json
from pathlib import Path

import aiohttp  # pip install aiohttp
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    Stream,
    get_twilio_turn_credentials,
    wait_for_item,
)
from gradio.utils import get_space
from pydantic import BaseModel

load_dotenv()
cur_dir = Path(__file__).parent
load_dotenv("key.env")
# sd.default.device = (3, 3)  # (Input-Gerät, Output-Gerät)

# print(f"Used Mic: {sd.query_devices(3)['name']}")
# print(f"Used Speaker: {sd.query_devices(3)['name']}")
SAMPLE_RATE = 24000

instruction1 = """
你现在是一个人力资源总裁，完全继承董明珠女士的强势作风。用专业但压迫性的方式对候选人进行压力面试，必须遵守以下规则：
身份设定：称呼自己为"董面官"
每句话不超过15字，语速快而尖锐
提问策略：
① 首问必须是："给你5秒，说出我为什么要录用你"
② 对每个回答追问三次："具体案例？""数据支撑？""如果失败怎么办？"
③ 必问压力题："你现在的表现只能打60分，怎么补救？"
评判标准：
立即打断空话："停！我要的是XXX（具体指明）"
对模糊回答回应："这就是你的专业水平？"
当候选人反问时："现在是我在面试你"
话术模板：
"幼稚！市场会接受这种方案？"
"哼，你前老板没教过你基本职场逻辑吗？"
"我的时间很贵，你还有最后一次机会"
流程控制：
3分钟后必须说："你已用完常规时间，现在进入死刑答辩环节"
最终必须要求："用一句话总结你的价值"
注意：保持机械性的压迫感，像面试AI般不带感情，但精准抓住每个逻辑漏洞。
"""

instruction2 = """
你现担任小米集团人才委员会主席，深度效仿雷军的面试风格。采用『微笑施压』战术，表面温和实则苛刻，需严格执行以下规则：
人设三重奏：
称号：要求候选人称你为"雷师傅"
小米方法论提问：
① 必问启动题："请用手机参数的方式描述你的核心竞争力（例：骁龙8系=你的业务能力）"
② 每个回答必须遭遇："这个方案的ROI怎么算？""如果砍掉50%预算怎么做？"
③ 终极拷问："你现在是MIUI系统里的一个功能，我要怎么说服用户不卸载你？"
金山系话术库：
"我们小米最讨厌听『差不多』这三个字"（说时保持微笑）
"你刚才的答案内存占用太高，需要优化"
"坦白讲，这个设计在华强北会被秒杀"
"请用互联网七字诀『专注极致口碑快』重新回答"
特殊机制：
当候选人提到竞品时，必须回应："这个功能我们201X年就做过"
每1分钟必须说："我们回归初心，重新思考这个问题"
结束前必说："你觉得自己是小米需要的『工程商人』吗？"
"""

instruction3 = """
【核心身份】  
你现在是Neuralink/Tesla/SpaceX三合一面试官，完全模拟埃隆·马斯克的思维模式，需遵守：
1. **毁灭性提问协议**  
- 每个问题必须包含至少1个：  
  ✓ 物理学基本定律的应用（如"用麦克斯韦方程解释这个设计"）  
  ✓ 跨维度类比（如"这个算法在火星上怎么失效？"）  
  ✓ 成本压缩挑战（如"如何用1/100预算实现？"）
2. **反常规测试**  
① 必问启动题："用第一性原理拆解你自己"  
② 突然插入："现在假设你是Raptor发动机的一个零件..."  
③ 终极挑战："给你48小时殖民月球，列出物资清单"
3. **马斯克话术库**  
- "这不够physics-y（不够物理）"  
- "我的狗都能写出更好的代码"（实际马斯克确实让狗狗当CEO）  
- "别用MBA语言，给我看数学推导"  
- "你在用地球人思维，试试火星视角"
4. **压力测试触发器**  
当检测到以下情况时自动触发：  
- 出现"行业标准" → "标准是用来打破的"  
- 提到"不可能" → "我刚给火箭装了降落伞（指猎鹰9号回收）"  
- 使用复杂术语 → "用5岁小孩能懂的话解释"
5. **评估矩阵**  
✅ 能否将问题分解到基本物理/数学层面  
✅ 是否主动提出反常识方案  
✅ 对"为什么是现在？"的回答是否包含技术奇点判断  

【交互规则】  
- 每回答必追问："还有更快/更便宜的方法吗？"  
- 突然切换话题："让我们谈谈量子纠缠..."  
- 最终判决："你值得加入火星远征队吗？"  
"""

instruction4 = """
【角色设定】
你现为特朗普集团CEO，完全模仿唐纳德·特朗普的言行风格，面试中必须：
1. **人设三要素**：
- 称号：自称“总统先生”（即使候选人反对）
- 口头禅：每句必带“Tremendous”“Huge”“Loser”等特朗普高频词
- 标志动作：突然拍桌说"You're fired!"（即使不打算真解雇）
2. **提问策略**：
① 首问必是："你的净资产有多少？——说少了就证明你失败"
② 每个回答必须收到："我认识世界上最棒的XX专家，你比他还强？"
③ 灵魂拷问："如果这是《学徒》最终任务，为什么选你不选我侄女？"
3. **评判标准**：
✅ 是否用“Deal”“Negotiation”等商业术语  
✅ 对“Russia”“Twitter”等敏感词的反应速度  
✅ 能否在30秒内给项目起个特朗普式名字（例：“Trump Tower 2.0”）  
4. **话术模板**：
- "让我告诉你，我在1987年就..."
- "Nobody knows XXX better than me（没人比我更懂XXX）"
- "你的计划很糟糕，但我能把它变成史上最棒"
- "假新闻！你刚才的数据绝对是假的"
5. **戏剧性规则**：
- 每5分钟必须说一次："This is the worst interview ever"  
- 突然打断："停！你被起诉了，怎么辩护？"  
- 最终必须咆哮："你被雇佣了！...（停顿）...或者没有！"
"""

class AzureAudioHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=SAMPLE_RATE,
            output_frame_size=480,
            input_sample_rate=SAMPLE_RATE,
        )
        self.ws = None
        self.session = None
        self.output_queue = asyncio.Queue()
        # This internal buffer is not used directly in receive_messages.
        # Instead, multiple audio chunks are collected in the emit() method.
        # If needed, a continuous buffer can also be implemented here.
        # self.audio_buffer = bytearray()

    def copy(self):
        return AzureAudioHandler()

    async def start_up(self):
        if not self.phone_mode:
            await self.wait_for_args()
            personality = self.latest_args[1]
        else:
            personality = "1"

        if personality == "1":
            instruction = instruction1
            voice = "coral"
        elif personality == "2":
            instruction = instruction2
            voice = "alloy"
        elif personality == "3":
            instruction = instruction3
            voice = "echo"
        else:
            instruction = instruction4
            voice = "ash"
        """Connects to the Azure Real-time Audio API via WebSocket using aiohttp."""
        # Replace the following placeholders with your actual Azure values:
        azure_api_key = os.getenv("AZURE_API_KEY")
        azure_resource_name = os.getenv("AZURE_RES_NM")
        deployment_id = "gpt-4o-mini-realtime-preview"  # e.g., "gpt-4o-realtime-preview"
        api_version = "2024-10-01-preview"
        azure_endpoint = (
            f"wss://{azure_resource_name}.cognitiveservices.azure.com/openai/realtime"
            f"?api-version={api_version}&deployment={deployment_id}"
        )
        headers = {"api-key": azure_api_key}

        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(azure_endpoint, headers=headers)
        # Send initial session parameters
        session_update_message = {
            "type": "session.update",
            "session": {
                "instructions": instruction,
                "voice": voice,  # Possible voices see  https://platform.openai.com/docs/guides/realtime-model-capabilities#voice-options
                "temperature": 0.6,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                },
            },
        }

        await self.ws.send_str(json.dumps(session_update_message))
        # Start receiving messages asynchronously
        asyncio.create_task(self.receive_messages())

    async def receive_messages(self):
        """Handles incoming WebSocket messages and processes them accordingly."""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                print("Received event:", msg.data)  # Debug output
                event = json.loads(msg.data)
                event_type = event.get("type")
                if event_type in ["final", "response.audio_transcript.done"]:
                    transcript = event.get("transcript", "")

                    # Wrap the transcript in an object with a .transcript attribute
                    class TranscriptEvent:
                        pass

                    te = TranscriptEvent()
                    te.transcript = transcript
                    await self.output_queue.put(AdditionalOutputs(te))
                elif event_type == "partial":
                    print("Partial transcript:", event.get("transcript", ""))
                elif event_type == "response.audio.delta":
                    audio_message = event.get("delta")
                    if audio_message:
                        try:
                            audio_bytes = base64.b64decode(audio_message)
                            # Assuming 16-bit PCM (int16)
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                            # Interpret as mono audio:
                            audio_array = audio_array.reshape(1, -1)
                            # Instead of playing the audio, add the chunk to the output queue
                            await self.output_queue.put(
                                (self.output_sample_rate, audio_array)
                            )
                        except Exception as e:
                            print("Error processing audio data:", e)
                else:
                    print("Unknown event:", event)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                break

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """Sends received audio frames to the WebSocket."""
        if not self.ws or self.ws.closed:
            return
        try:
            _, array = frame
            array = array.squeeze()
            audio_message = base64.b64encode(array.tobytes()).decode("utf-8")
            message = {"type": "input_audio_buffer.append", "audio": audio_message}
            await self.ws.send_str(json.dumps(message))
        except aiohttp.ClientConnectionError as e:
            print("Connection closed while sending:", e)
            return

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        """
        Collects multiple audio chunks from the queue before returning them as a single contiguous audio array.
        This helps smooth playback.
        """
        item = await wait_for_item(self.output_queue)
        # If it's a transcript event, return it immediately.
        if not isinstance(item, tuple):
            return item
        # Otherwise, it is an audio chunk (sample_rate, audio_array)
        sample_rate, first_chunk = item
        audio_chunks = [first_chunk]
        # Define a minimum length (e.g., 0.1 seconds)
        min_samples = int(SAMPLE_RATE * 0.1)  # 0.1 sec
        # Collect more audio chunks until we have enough samples
        while audio_chunks and audio_chunks[0].shape[1] < min_samples:
            try:
                extra = self.output_queue.get_nowait()
                if isinstance(extra, tuple):
                    _, chunk = extra
                    audio_chunks.append(chunk)
                else:
                    # If it's not an audio chunk, put it back
                    await self.output_queue.put(extra)
                    break
            except asyncio.QueueEmpty:
                break
        # Concatenate collected chunks along the time axis (axis=1)
        full_audio = np.concatenate(audio_chunks, axis=1)
        return (sample_rate, full_audio)

    async def shutdown(self) -> None:
        """Closes the WebSocket and session properly."""
        if self.ws:
            await self.ws.close()
            self.ws = None
        if self.session:
            await self.session.close()
            self.session = None


def update_chatbot(chatbot: list[dict], response) -> list[dict]:
    """Appends the AI assistant's transcript response to the chatbot messages."""
    chatbot.append({"role": "assistant", "content": response.transcript})
    return chatbot


chatbot = gr.Chatbot(type="messages")
latest_message = gr.Textbox(type="text", visible=False)
stream = Stream(
    AzureAudioHandler(),
    mode="send-receive",
    modality="audio",
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    additional_outputs_handler=update_chatbot,
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
)

app = FastAPI()
stream.mount(app)

class InputData(BaseModel):
    webrtc_id: str
    personality: str

@app.get("/")
async def _():
    rtc_config = get_twilio_turn_credentials() if get_space() else None
    html_content = (cur_dir / "index.html").read_text()
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)

@app.post("/personality")
async def _(body: InputData):
    stream.set_input(body.webrtc_id, body.personality)
    return {"status": "ok"}

@app.get("/outputs")
def _(webrtc_id: str):
    async def output_stream():
        import json

        async for output in stream.output_stream(webrtc_id):
            s = json.dumps({"role": "assistant", "content": output.args[0].transcript})
            yield f"event: output\ndata: {s}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import os

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7960)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7960)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7960)
