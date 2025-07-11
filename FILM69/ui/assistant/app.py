from flet import *
try:
    from .ui import Ui_app
    from .langflow_api import LangflowAPI
except:
    from ui import Ui_app
    from langflow_api import LangflowAPI
import pyaudio
import numpy as np
import random
import time,json
import threading
import os
import site
from pathlib import Path
import uuid

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
        self.bot_name="เสียวอู่"
        
        print("load model success")

        self.chat.user_name="User"
        self.chat.model_name=self.bot_name  

        self.config=None
        self.session_id=str(uuid.uuid4())
        # self.session_id="123"

        self.url=TextField(label="URL")
        self.flow_id=TextField(label="Flow ID")
        self.api_key=TextField(label="API KEY",password=True, can_reveal_password=True)
        self.save_config_btn=Button("Save",expand=True,on_click=lambda e: self.save_config())
        
        self.load_config()
        if self.url.value!="" and self.flow_id!="" and self.api_key!="":
            self.get_history_chat()
        
        
        new_tab=[
            Tab(
                tab_content=Icon(Icons.HISTORY),
                
            ),
            Tab(
                tab_content=Icon(Icons.SETTINGS),
                content=Container(
                    content=Column([
                        Text(),
                        self.url,
                        self.flow_id,
                        self.api_key,
                        Row([self.save_config_btn])
                    ]), alignment=alignment.center
                ),
            ),
            ]
        
        for i in new_tab:self.tabs.tabs.append(i)
        
        if self.url.value!="" and self.flow_id!="" and self.api_key!="":
            self.update_history_tab()
       
    
    def update_history_tab(self):
        self.tabs.tabs[1].content=self.chat_session()
        self.page.update()

    def chat_session(self):
        data=self.agent.get_messages(all=True)
        seen = set()
        unique = [d for d in data if d['session_id'] not in seen and not seen.add(d['session_id'])]

        x=ListView([
                CupertinoButton(i["input"],on_click=lambda e:self.select_chat(i["session_id"]))
        for i in unique])
        
        return x

    def load_config(self):
        path_f=self.path_config()
        path=f"{path_f}/config.json"
        if "config.json" not in os.listdir(path_f):
            data={"url":"","flow_id":"","api_key":""}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        with open(path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            self.url.value=self.config["url"]
            self.flow_id.value=self.config["flow_id"]
            self.api_key.value=self.config["api_key"]

    def save_config(self):
        path=self.path_config()
        path=f"{path}/config.json"
        data={"url":self.url.value,"flow_id":self.flow_id.value,"api_key":self.api_key.value}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        self.dialog()
        
    
    def dialog(self,title="บันทึกสำเร็จ",content="",yes_btn="ตกลง",icon=Icons.VERIFIED):
        dlg_modal = AlertDialog(
            modal=True,
            # icon=icon,
            title=Text(title),
            content=Text(content),
            actions=[
                TextButton(yes_btn, on_click=lambda e: self.page.close(dlg_modal)),
            ],
            actions_alignment=MainAxisAlignment.END,
        )
        self.page.open(dlg_modal)

    def path_config(self):
        site_packages = next(p for p in site.getsitepackages() if Path(p).exists())
        assistant_path = Path(site_packages)/ "Lib" / "site-packages" / "FILM69" / "ui" / "assistant"
        if assistant_path.exists():
            # print(f"✅ พบโฟลเดอร์อยู่แล้ว: {assistant_path}")
            ...
        else:
            assistant_path.mkdir(parents=True, exist_ok=True)
            # print(f"🆕 สร้างโฟลเดอร์ใหม่เรียบร้อย: {assistant_path}")
        
        return assistant_path

    def select_chat(self,session_id):
        self.tabs.selected_index=0
        self.session_id=session_id
        self.get_history_chat()
        self.page.update()

    def get_history_chat(self):
        self.agent=LangflowAPI(
            url=self.url.value,
            flow_id=self.flow_id.value,
            session_id=self.session_id,
            api_key=self.api_key.value
        )

        for char_js in self.agent.get_messages():
            self.chat.message_input.value=char_js["input"]
            self.chat.send_message()
            current_text = char_js["output"]
            tool = char_js.get("tool")
            js = {"Input": char_js["input"]}
            if tool!=None:
                index=0
                for i in tool:
                    tool_name = i.get("name")
                    js[f"Executed {tool_name}{' '*index}"] = f'''
Input:
```json
{json.dumps(i.get("input", {}), indent=2,ensure_ascii=False)}
```

Output:

```json
{json.dumps(i.get("output", {}), indent=2,ensure_ascii=False)}
```'''
                    index+=1
            js["Output"] = current_text


            self.chat.update_status("Finished", 0, js)
            self.chat.update_output_text(current_text, False)
            self.page.update()


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
                            i.bgcolor="#00ff00"
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
                        self.page.update()
                        self.fn(None)
                        
                elif self.mic.icon==Icons.MIC:
                    self.chat.message_input.value = text.replace(self.bot_name,"").replace(" ","")
                    self.page.update()
                    self.fn(None)
                
    def fn(self,e):
        self.chat.send_message()
        stop=False
        start_time = time.time()
        current_text = ""
        
        def time_update():
                while not stop:
                    elapsed = time.time() - start_time
                    self.chat.message_card.time_text.value = f"{elapsed :.1f}s"
                    self.page.update()
            
        threading.Thread(target=time_update).start()
        

        js = {"Input": self.chat.input_text}
        elapsed = time.time() - start_time
        self.chat.update_status("Processing", elapsed, js)
        for char_js in self.agent.chat(self.chat.input_text):
            current_text = char_js["output"]
            tool = char_js.get("tool")

            index=0
            for i in tool:
                tool_name = i.get("name")
                
                if tool_name:
                    js[f"Executed {tool_name}{' '*index}"] = f'''
Input:
```json
{json.dumps(i.get("input", {}), indent=2,ensure_ascii=False)}
```

Output:

```json
{json.dumps(i.get("output", {}), indent=2,ensure_ascii=False)}
```'''
                    index+=1
                if current_text=="":
                    self.chat.update_status(f"Executed {tool_name}", elapsed, js)

                
            if current_text !="":
                js["Output"] = current_text
                self.chat.update_status("Generating", elapsed, js)

            self.chat.update_output_text(current_text, True)
            self.page.update()
            

            
        # Finished
        stop=True
        final_time = time.time() - start_time
        self.chat.update_status("Finished", final_time, js)
        self.chat.update_output_text(current_text, False)
        self.page.update()

def run(page:Page):
    App(page)

def main():
    app(run)

if __name__=="__main__":
    # app(main,port=7860,view=AppView.FLET_APP_WEB)
    app(run)

