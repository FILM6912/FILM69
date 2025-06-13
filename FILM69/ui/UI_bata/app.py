from flet import *
from ui import Ui_app
from llm import llm
import pyaudio
import numpy as np
import random
import time

on_whisper=False
if on_whisper:
    from FILM69.stt import Whisper

class App(Ui_app):
    def __init__(self,page:Page):
        super().__init__(page)
        
        if on_whisper:
            self.whisper=Whisper()
            self.whisper.load_model("FILM6912/whisper-small-thai")
        
        self.mic_off=False
        self.bot_name="คอมพิวเตอร์"
        
        print("load model success")
        
        self.chat.user_name="ggg"
        self.chat.model_name=self.bot_name

                
    def audio_capture(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,channels=self.CHANNELS,rate=self.RATE,input=True,frames_per_buffer=self.CHUNK)
        try:
            while True:
                silence_counter = 0
                detect = False
                frames = []
                # if self.mic.icon==icons.MIC_OFF: break
                if self.mic_off: break
                while True:
                    # if self.mic.icon==icons.MIC_OFF: break
                    if self.mic_off: break
                    data = stream.read(self.CHUNK)
                    np_data = np.frombuffer(data, dtype=np.int16)
                    frames.append(data)
                    if np.abs(np_data).mean() > self.THRESHOLD:
                        self.page.update()
                        for i in self.voice_status.controls:
                            # i.height=random.randint(0,int(np.abs(np_data).mean()))/2
                            i.height=random.randint(0,int(np.abs(np_data).mean()))
                            i.update()
                        detect = True
                        silence_counter = 0
                    else:
                        silence_counter += 1
                        for i in self.voice_status.controls:
                            i.height=1
                            i.update()

                    if silence_counter > int(self.SILENCE_DURATION * self.RATE / self.CHUNK) and detect:
                        print("กำลังประมวลผลเสียง....")
                        # self.page.update()
                        yield b''.join(frames)
                        break
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            
    def update_voive_status(self):
        self.CHUNK = 1024 * 5
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.THRESHOLD = 30
        self.SILENCE_DURATION = 2
        for frame in self.audio_capture():
            audio_data = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            if on_whisper:
                text=self.whisper.predict(audio_data)["text"]
                print(text)
                
                if self.mic.icon==Icons.MIC_OFF:
                    if self.bot_name in text:
                        self.chat.message_input.value = text.replace(self.bot_name,"").replace(" ","")
                        self.chat.send_message()
                elif self.mic.icon==Icons.MIC:...
                    # self.new_message.value = text
                    # self.page.update()
                    # self.send_message_click(None)
                
    def fn(self,e):
        responses = [
            "สวัสดีครับ มีอะไรให้ช่วยบ้างไหมครับ",
            "ผมชื่อ AI Assistant ยินดีที่ได้รู้จักครับ",
            "ขอบคุณสำหรับคำถามครับ ผมจะช่วยเหลือคุณให้ดีที่สุด",
            "นั่นเป็นคำถามที่น่าสนใจมากครับ ให้ผมอธิบายให้ฟังนะครับ",
        ]
        
        self.chat.send_message()
        
        
        start_time = time.time()
        response_text = random.choice(responses)
    
        # processing_time = random.uniform(1.0, 2.0)
        # while time.time() - start_time < processing_time:
        #     elapsed = time.time() - start_time
        #     self.chat.update_status("Processing", elapsed, {
        #         "Input": self.chat.message_input.value,
        #     })
        #     self.page.update()
        #     time.sleep(0.1)
        
        current_text = ""
        print(self.chat.input_text)
        for char in llm.stream(self.chat.input_text):
            current_text += char.content
            elapsed = time.time() - start_time
            self.chat.update_status("Generating", elapsed, {
                "Input": self.chat.input_text,
                "Output": current_text
            })
            self.chat.update_output_text(current_text, True)
            self.page.update()
            
            # time.sleep(0.01)
        
        # Finished
        final_time = time.time() - start_time
        self.chat.update_status("Finished", final_time, {
            "Input": self.chat.input_text,
            "Output": current_text
        })
        self.chat.update_output_text(current_text, False)
        self.page.update()

def run(page:Page):
    App(page)

def main():
    app(run)

if __name__=="__main__":
    # app(main,port=7860,view=AppView.FLET_APP_WEB)
    app(run)