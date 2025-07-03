import flet as ft
import time
import random
from typing import Union,Literal
try:
    from .message_card import MessageCard
except:
    from widget.message_card import MessageCard

class Chat(ft.Container):
    def __init__(self, page: ft.Page):
        super().__init__()
        self.page = page
        self.fn=None
        self.setup_ui()
        self.setup_properties()
        self.input_text=""
        self.chat_history=[]
        self.user_name="User"
        self.model_name="XiaoWo"
    
    def load_fn(self,fn):
        self.fn = fn
        self.send_button.on_click=self.fn
    
    def setup_properties(self):
        """Setup container properties"""
        self.expand = True
        self.padding = 0
        self.bgcolor = "#1a1a1a"
    
    def setup_ui(self):
        """Initialize UI components"""
        # Create message input field
        self.message_input = ft.TextField(
            hint_text="Send a message...",
            border=ft.InputBorder.OUTLINE,
            border_color="#404040",
            bgcolor="#2a2a2a",
            color="#ffffff",
            hint_style=ft.TextStyle(color="#888888"),
            expand=True,
            multiline=True,
            min_lines=1,
            max_lines=3,
            on_submit=self.fn,
        )
        
        # Chat messages container
        self.chat_container = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=15,
        )
        
        # Send button
        self.send_button = ft.IconButton(
            icon=ft.Icons.SEND,
            icon_color="#888888",
            # bgcolor="#8b5cf620",
            on_click=self.fn,
        )
        
        # Input area
        self.input_area = ft.Container(
            content=ft.Row([
                ft.IconButton(
                    icon=ft.Icons.ATTACH_FILE,
                    icon_color="#888888",
                ),
                self.message_input,
                ft.IconButton(
                    icon=ft.Icons.MIC,
                    icon_color="#888888",
                ),
                self.send_button,
            ], spacing=5),
            bgcolor="#2a2a2a",
            padding=ft.padding.all(15),
            border=ft.border.only(top=ft.border.BorderSide(1, "#404040")),
            # border_radius=25
        )
        # Set main content
        self.content = ft.Column([
            ft.Container(
                content=self.chat_container,
                expand=True,
                padding=ft.padding.symmetric(vertical=10),
            ),
            self.input_area,
        ], spacing=0, expand=True)
        
        # Add sample messages
        # self.add_sample_messages()
        
        # Setup keyboard events
        self.setup_keyboard_events()

    
    def setup_keyboard_events(self):
        """Setup keyboard event handling"""
        def on_keyboard(e: ft.KeyboardEvent):
            if e.key == "Enter" and not e.shift:
                self.fn(None)
        
        self.page.on_keyboard_event = on_keyboard

    def send_message(self):
        self.message_card = MessageCard(
            user_name=self.user_name,
            model_name=self.model_name,
            input_text=self.message_input.value
        )
        self.chat_container.controls.append(self.message_card.get_card())
        self.input_text=self.message_input.value
        self.message_input.value = ""
        self.page.update()
        self.chat_container.scroll_to(offset=-1, duration=300)
        self.chat_history.append({
            "role": "user",
            "content":[
                {"type": "text", "text": self.input_text},
            ]
        })
        
        self.chat_history.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": ""},
            ]
        })
        
        
    def update_status(self,status,elapsed_time,current_data:dict):
        self.message_card.update_status(
            status, elapsed_time, current_data)
        self.chat_container.scroll_to(offset=-1, duration=300)
        
        
    def update_output_text(self,text,is_generating=False):
        self.message_card.update_output_text(text, is_generating)
        self.chat_container.scroll_to(offset=-1, duration=300)
        self.chat_history[-1]["content"][0]["text"]=text
        
    def get_chat(self):
        return self.chat_history
    

def main(page: ft.Page):
    page.title = "Streaming Chat Interface"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#1a1a1a"
    page.padding = 0
    
    # Create and add chat interface
    chat = Chat(page)
    chat.user_name="User"
    chat.model_name="XiaoWo"
    
    def fn(e):
        responses = [
            "สวัสดีครับ มีอะไรให้ช่วยบ้างไหมครับ",
            "ผมชื่อ AI Assistant ยินดีที่ได้รู้จักครับ",
            "ขอบคุณสำหรับคำถามครับ ผมจะช่วยเหลือคุณให้ดีที่สุด",
            "นั่นเป็นคำถามที่น่าสนใจมากครับ ให้ผมอธิบายให้ฟังนะครับ",
        ]
        
        chat.send_message()
        start_time = time.time()
        response_text = random.choice(responses)
        
        chat.chat_container.scroll_to(offset=-1, duration=300)
    
        processing_time = random.uniform(1.0, 2.0)
        while time.time() - start_time < processing_time:
            elapsed = time.time() - start_time
            chat.update_status("Processing", elapsed, {
                "Input": chat.message_input.value,
            })
            page.update()
            time.sleep(0.1)
        
        current_text = ""
        for i, char in enumerate(response_text):
            current_text += char
            elapsed = time.time() - start_time
            chat.update_status("Generating", elapsed, {
                "Input": chat.message_input.value,
                "Tools":current_text,
                "Output": current_text
            })
            chat.update_output_text(current_text, True)
            page.update()
            
            time.sleep(0.01)
        
        # Finished
        final_time = time.time() - start_time
        chat.update_status("Finished", final_time, {
            "Input": chat.message_input.value,
            "Tools":current_text,
            "Output": current_text
        })
        chat.update_output_text(current_text, False)
        page.update()
        
    chat.load_fn(fn)
    
    page.add(chat,ft.Button("get",on_click=lambda e: print(chat.get_chat())))

if __name__ == "__main__":
    ft.app(target=main)