import flet as ft
import time

class MessageContainer(ft.Container):
    def __init__(
        self,
        body,
        bgcolor=ft.Colors.BLACK87,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        code_theme=ft.MarkdownCodeTheme.MONOKAI,
        code_style_sheet=ft.MarkdownStyleSheet.block_spacing
        ):
        
        super().__init__()
        self.content = ft.Column(
            controls=[
                ft.Markdown(
                    body,
                    selectable=True,
                    extension_set=extension_set,
                    on_tap_link=lambda e: self.page.launch_url(e.data),
                    code_theme=code_theme,
                    code_style_sheet=code_style_sheet
                ),
            ],
        )
        self.border = ft.border.all(1, ft.Colors.BLACK)
        self.border_radius = ft.border_radius.all(10)
        self.bgcolor = bgcolor
        self.padding = 10
        self.expand = True
        self.expand_loose = True

class Message(ft.Row):
    def __init__(
        self,
        text, 
        name, 
        user_or_ai="user",
        bgcolor=ft.Colors.BLACK87,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        code_theme=ft.MarkdownCodeTheme.MONOKAI,
        code_style_sheet=ft.MarkdownStyleSheet.block_spacing):
        super().__init__()
        self.alignment = (
            ft.MainAxisAlignment.START if user_or_ai == "ai" else ft.MainAxisAlignment.END
        )
        self.vertical_alignment = ft.CrossAxisAlignment.START
        self.user_or_ai = user_or_ai
        if user_or_ai == "ai":
            self.controls = [
                ft.CircleAvatar(bgcolor="blue", content=ft.Text(name, color="white")),
                MessageContainer(
                    body=text,
                    bgcolor=bgcolor,
                    extension_set=extension_set,
                    code_theme=code_theme,
                    code_style_sheet=code_style_sheet),
            ]
        else:
            self.controls = [
                MessageContainer(
                    body=text,
                    bgcolor=bgcolor,
                    extension_set=extension_set,
                    code_theme=code_theme,
                    code_style_sheet=code_style_sheet),
                ft.CircleAvatar(bgcolor="green", content=ft.Text(name, color="white")),
            ]

class Chat(ft.ListView):
    def __init__(
        self,
        bgcolor=ft.Colors.BLACK87,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        code_theme=ft.MarkdownCodeTheme.MONOKAI,
        code_style_sheet=ft.MarkdownStyleSheet.block_spacing
        
        ):
        super().__init__()
        self.padding = 10
        self.spacing = 10
        self.auto_scroll = True
        self.expand = True  # สำคัญมากให้ขยายเต็มพื้นที่
        self.controls = []
        self.bg=bgcolor
        self.extension_set=extension_set
        self.code_theme=code_theme
        self.code_style_sheet=code_style_sheet



    def add_message(self, name, text, user_or_ai="user"):
        self.controls.append(Message(
            text, 
            name, 
            user_or_ai, 
            bgcolor=self.bg,
            extension_set=self.extension_set,
            code_theme=self.code_theme,
            code_style_sheet=self.code_style_sheet))
        self.update()
        
    def add_stream_message(self, text):
        self.controls[-1].controls[1].content.controls[0].value=text
        self.update()
        

    def get_message(self):
        messages = []  # reset ใหม่ทุกครั้ง
        for i in self.controls:
            if i.user_or_ai == "ai":
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": i.controls[1].content.controls[0].value},
                    ]
                })
            elif i.user_or_ai == "user":
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": i.controls[0].content.controls[0].value},
                    ]
                })
            elif i.user_or_ai == "system":
                messages.append({
                    "role": "system",
                    "content":[
                        {"type": "text", "text": i.controls[0].content.controls[0].value},
                    ]
                })
            
        return messages

def main(page: ft.Page):
    chat = Chat(
        # bgcolor=ft.Colors.BROWN,
        # code_theme=ft.MarkdownCodeTheme.DARK,
        )

    def get(e):
        print(chat.get_message())

    page.add(
        ft.Column(
            [
                chat,
                ft.ElevatedButton("get", on_click=get),
            ],
            expand=True
        )
    )

    # ตัวอย่างข้อความทดสอบ
    chat.add_message("User", "แปลภาษาให้หน่อย", "user")
    
    text="""```sh
pip install "git+https://github.com/watcharaphon6912/film69.git@v0.4.7#egg=film69[all]"
````

```python
from FILM69.llm import FastAutoModel
from PIL import Image

image = Image.open("image.jpg")

model = FastAutoModel(
    "FILM6912/Llama-3.2-11B-Vision-Instruct",
    device_map="cuda",
    load_in_4bit=True,
)
```"""
    chat.add_message("Ai", "start", "ai")
    t=""
    for i in text:
        t+=i
        chat.add_stream_message(t)
        time.sleep(0.001)


    page.window.width = 393
    page.window.height = 600
    page.window.always_on_top = False

if __name__ == "__main__":
    ft.app(target=main)
