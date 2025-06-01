from flet import *
from FILM69.ui import Timer,Chat
import time
import random
import keyboard as kb

text_test="""
### install
```sh
pip install git+https://github.com/watcharaphon6912/film69.git#egg=film69[all]
```
```sh
pip install git+https://github.com/watcharaphon6912/film69.git#egg=film69[rag]
```

### example
#### LLM
```python
from film69.ml.model import LLMModel
model=LLMModel(
    "scb10x/typhoon-7b-instruct-02-19-2024",
    device_map="cuda",
    load_in_4bit=True,
    # load_in_8bit=True,
    # low_cpu_mem_usage = True
)
for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")
print(model.generate("สวัสดี",max_new_tokens=200))
```
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
        self.chat = Chat()
        self.new_message = TextField(hint_text="ป้อนคำสั่ง",autofocus=True,shift_enter=True,max_lines=20,filled=True,expand=True,on_submit=self.send_message_click)
        self.send_message = IconButton(icon=Icons.SEND_ROUNDED,tooltip="ส่ง",scale=2,on_click=self.send_message_click)

        controls=Row([
            Switch(value=True,label="แสดงอนิเมชั่น"),
            Switch(value=False,label="เสียงตอบกลับ")],alignment=MainAxisAlignment.CENTER)

        self.voice_status=Row([Container(height=35,expand=True,bgcolor="#ff6600",border_radius=10,animate=Animation(500, AnimationCurve.EASE_OUT))for i in range(10)])
        self.stop_button=ElevatedButton("Stop",icon=Icons.STOP,icon_color="red",visible=False,on_click=lambda e: setattr(self, 'stop_gen', True))
        
        st=Stack([self.chat,self.stop_button],alignment=alignment.bottom_center,height=635)
        self.page_chat=Column([controls,st,Row([self.new_message,self.send_message,])],expand=True)
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
    def get_initials(self, user_name: str):
        if user_name:return user_name[:1].capitalize()
        else:return "F" 

    def get_avatar_color(self, user_name: str):
        Colors_lookup = [
            Colors.ON_ERROR_CONTAINER,
            Colors.BLUE,
            Colors.BROWN,
            Colors.CYAN,
            Colors.GREEN,
            Colors.INDIGO,
            Colors.LIME,
            Colors.ON_PRIMARY,
            Colors.PINK,
            Colors.PURPLE,
            Colors.RED,
            Colors.TEAL,
            Colors.YELLOW,
        ]
        return Colors_lookup[hash(user_name) % len(Colors_lookup)]
    
    def send_message_click(self,e):
        text = self.new_message.value.strip()
        if text:
            self.chat.add_message("User", text, "user")
            # จำลองคำตอบจาก AI
            self.chat.add_message("AI", "...", "ai")
            self.new_message.value = ""
            self.page.update()
            t=""
            for i in text_test:
                t+=i
                self.chat.add_stream_message(t)
                time.sleep(0.01) 

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

def main(page:Page):
    Ui_app(page)

if __name__=="__main__":
    # app(main,port=7860,view=AppView.FLET_APP_WEB)
    app(main)