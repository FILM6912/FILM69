from flet import *
from FILM69.ui import Timer
try:
    from .widget import Chat
except:
    from widget import Chat
import time
import random
import keyboard as kb

text_test="""
# example
#### FastAutoModel
```python
from FILM69.llm import FastAutoModel
from PIL import Image

image=Image.open("image.jpg")

model=FastAutoModel(
    "FILM6912/Llama-3.2-11B-Vision-Instruct",
    device_map="cuda",
    load_in_4bit=True,
    )

for i in model.generate("คุณเห็นอะไรในรูป",image,stream=True):
    print(i,end="")

for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")

print(model.generate("คุณเห็นอะไรในรูป",image,max_new_tokens=200))
print(model.generate("สวัสดี",max_new_tokens=200))


#####################################################################################

model=FastAutoModel(
    "FILM6912/Llama-3.2-1B-Instruct",
    device_map="cuda",
    load_in_4bit=True,
    )

for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")

print(model.generate("สวัสดี",max_new_tokens=200))

```



#### FastVLLM
"""

class Ui_app:
    def __init__(self,page:Page):
        self.page=page
        self.page.window.width=300
        # self.page.window.width=800
        
        self.page.window.height=60
        
        self.page.bgcolor=Colors.TRANSPARENT
        self.page.window.frameless=True
        # self.page.window.bgcolor=Colors.BLACK
        self.page.window.bgcolor=Colors.TRANSPARENT
        
        self.page.window.resizable=False
        self.page.window.always_on_top=True
        self.page.window.alignment=alignment.top_center
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.user_message = ""
        self.user_name="Film"
        self.bot_name="XiaoXi"   

        self.stop_gen=False
        self.page.update()
        self.chat = Chat(self.page)
        self.chat.border_radius=10
        self.chat.load_fn(self.fn)
        
        # self.new_message = TextField(hint_text="ป้อนคำสั่ง",autofocus=True,shift_enter=True,max_lines=20,filled=True,expand=True,on_submit=self.send_message_click)
        # self.send_message = IconButton(icon=Icons.SEND_ROUNDED,tooltip="ส่ง",scale=2,on_click=self.send_message_click)

        controls=Row([
            Switch(value=True,label="แสดงอนิเมชั่น"),
            Switch(value=False,label="เสียงตอบกลับ")],alignment=MainAxisAlignment.CENTER)

        self.voice_status=Row([Container(height=35,expand=True,bgcolor="#ff6600",border_radius=10,animate=Animation(500, AnimationCurve.EASE_OUT))for i in range(10)])
        self.stop_button=ElevatedButton("Stop",icon=Icons.STOP,icon_color="red",visible=False,on_click=lambda e: setattr(self, 'stop_gen', True))
        
        self.st=Stack([self.chat,self.stop_button],alignment=alignment.bottom_center,height=700)

        self.tabs=Tabs(
            selected_index=0,
            animation_duration=300,
            tab_alignment=TabAlignment.CENTER,
            tabs=[
                    Tab(
                        tab_content=Icon(Icons.CHAT),
                        content=Container(
                            content=self.st, alignment=alignment.center
                        ),
                    ),
                ],
                expand=1,
            )
        self.page_chat=self.tabs

        # self.page_chat=Column([controls,self.st],expand=True)
        # self.page_chat=Column([controls,self.chat,Row([self.new_message,self.send_message,])],expand=True)

        self.menu=IconButton(icon=Icons.MENU,icon_size=30,icon_color=Colors.BLUE,on_click=self.open_or_close_chat,rotate=0,animate_rotation=Animation(300, AnimationCurve.EASE_IN_OUT_CIRC))
        self.mic=IconButton(icon=Icons.MIC_OFF,icon_size=30,icon_color="red",on_click=self.update_mic_status)
        
        self.con=Container(bgcolor="#494a49",height=self.page.window.height,width= self.page.window.width,alignment=alignment.bottom_center,content=self.page_chat)
        
        self.chat_page=Container(expand=True,bgcolor="#494a49",content=self.con,alignment=alignment.center,border_radius=10)
        
        self.bot_speak=[Container(height=190,expand=True,bgcolor="green",border_radius=50,animate=Animation(500, AnimationCurve.EASE_OUT))for i in range(10)]

        self.page.add(
            Timer(self.update_voive_status),
            # Timer(name="timer1", interval_s=0, callback=self.check_voice),
            self.chat_page,
            AppBar(title=Container(content=Row([
                    self.mic,
                    Container(content=self.voice_status,height=40,width=155,on_click=self.update_bot_speak_status),
                    WindowDragArea(self.menu,maximizable=False)
                    ],
                    alignment=MainAxisAlignment.CENTER,
                ),bgcolor="#000000",width=268,border_radius=50),
                center_title=True,
                bgcolor="transparent",
                ),
        )
        self.page.update()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    def expand_page(self):
        self.page.window.left -= 250
        self.page.window.width=800
        self.page.window.height += 800
        self.con.height=self.page.window.height-120
        self.con.width=self.page.window.width-35
    
    def shrink_page(self):
        self.page.window.left += 250
        self.page.window.width=300
        self.page.window.height -= 800
        self.con.height=self.page.window.height-120
        self.con.width=self.page.window.width-35
    
    def open_or_close_chat(self,e):
        self.con.content=self.page_chat
        self.chat_page.bgcolor="#494a49"
        self.con.bgcolor="#494a49"
        if self.menu.icon==Icons.MENU:
            self.menu.rotate=15.7
            self.menu.icon=Icons.MENU_OPEN
            self.expand_page()
        else:
            self.menu.rotate=0
            self.menu.icon=Icons.MENU
            self.shrink_page()
        self.page.update()

    def update_voive_status(self):
        for i in self.voice_status.controls:
            i.height=random.randint(1,35)
        self.page.update()
        time.sleep(0.5)


    def check_voice(self):
        while kb.is_pressed("x+v"):
            self.update_mic_status("")
            while kb.is_pressed("x+v"):...
        while kb.is_pressed("x+c"):
            self.open_or_close_chat("")
            while kb.is_pressed("x+c"):...

    def update_mic_status(self,e):
        if self.mic.icon==Icons.MIC:
            self.mic.icon=Icons.MIC_OFF
            self.mic.icon_color="red"
        else:
            self.mic.icon=Icons.MIC
            self.mic.icon_color="green"
        self.page.update()

    def update_bot_speak_status(self,e):
        if self.menu.icon==Icons.MENU:
            self.menu.icon=Icons.MENU_OPEN
            self.con.bgcolor=Colors.TRANSPARENT
            self.chat_page.bgcolor=Colors.TRANSPARENT
            self.con.alignment=alignment.center
            self.bot_speak_status=Container(Row(self.bot_speak),
                height=200,
                width=500
            )
            self.con.content=self.bot_speak_status
            self.expand_page()
        else:
            self.menu.icon=Icons.MENU
            self.con.content=self.page_chat
            self.shrink_page()
        
        self.page.update()
        
    def fn(self,e):
        responses = [
            "สวัสดีครับ มีอะไรให้ช่วยบ้างไหมครับ",
            "ผมชื่อ AI Assistant ยินดีที่ได้รู้จักครับ",
            "ขอบคุณสำหรับคำถามครับ ผมจะช่วยเหลือคุณให้ดีที่สุด",
            "นั่นเป็นคำถามที่น่าสนใจมากครับ ให้ผมอธิบายให้ฟังนะครับ",
        ]
        
        self.chat.send_message()
        
        start_time = time.time()
        # response_text = random.choice(responses)
        response_text = text_test
    
        processing_time = random.uniform(1.0, 2.0)
        while time.time() - start_time < processing_time:
            elapsed = time.time() - start_time
            self.chat.update_status("Processing", elapsed, {
                "Input": self.chat.message_input.value,
            })
            self.page.update()
            
            time.sleep(0.1)
        
        current_text = ""
        for i, char in enumerate(response_text):
            current_text += char
            elapsed = time.time() - start_time
            self.chat.update_status("Generating", elapsed, {
                "Input": self.chat.message_input.value,
                "Tools":"",
                "Output": current_text
            })
            self.chat.update_output_text(current_text, True)
            self.page.update()
            
            time.sleep(0.01)
        
        # Finished
        final_time = time.time() - start_time
        self.chat.update_status("Finished", final_time, {
            "Input": self.chat.message_input.value,
            "Tools":"",
            "Output": current_text
        })
        self.chat.update_output_text(current_text, False)
        self.page.update()
        


def main(page:Page):
    Ui_app(page)

if __name__=="__main__":
    # app(main,port=7860,view=AppView.FLET_APP_WEB)
    app(main)