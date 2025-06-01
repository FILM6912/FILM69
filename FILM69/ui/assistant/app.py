from flet import *
from .ui import Ui_app
from .llm import llm
from FILM69.stt import Whisper
import pyaudio
import numpy as np
import random

class App(Ui_app):
    def __init__(self,page:Page):
        super().__init__(page)
        self.whisper=Whisper()
        self.whisper.load_model("FILM6912/whisper-small-thai")
        
        self.mic_off=False
        self.bot_name="คอมพิวเตอร์"
        
        print("load model success")
    
    def send_message_click(self,e):
        text = self.new_message.value.strip()
        if text:
            self.chat.add_message("User", text, "user")
            # จำลองคำตอบจาก AI
            self.new_message.value = ""
            self.page.update()
            t=""
            mess=self.chat.get_message()
            his=[]
            for i in mess:
                if i["role"]=="user":
                    his.append(("user",i["content"]))
                else:
                    his.append(("ai",i["content"]))
            
            his.append(("user",text))
            self.chat.add_message("AI", "...", "ai")
            
            for i in llm.stream(his):
                t+=i.content
                self.chat.add_stream_message(t)
                
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
            text=self.whisper.predict(audio_data)["text"]
            print(text)
            
            if self.mic.icon==Icons.MIC_OFF:
                if self.bot_name in text:
                    self.new_message.value = text.replace(self.bot_name,"").replace(" ","")
                    if self.new_message.value !="":
                        self.page.update()
                        self.send_message_click(None)
            elif self.mic.icon==Icons.MIC:
                self.new_message.value = text
                self.page.update()
                self.send_message_click(None)

def run(page:Page):
    App(page)

def main():
    app(run)

if __name__=="__main__":
    # app(main,port=7860,view=AppView.FLET_APP_WEB)
    app(run)