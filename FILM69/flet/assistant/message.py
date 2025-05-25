import flet as ft

class MessageContainer(ft.Container):
    def __init__(self, body):
        super().__init__()
        self.content = ft.Column(
            controls=[
                ft.Markdown(
                    body,
                    selectable=True,
                    extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                    on_tap_link=lambda e: self.page.launch_url(e.data),
                ),
            ],
        )
        self.border = ft.border.all(1, ft.Colors.BLACK)
        self.border_radius = ft.border_radius.all(10)
        self.bgcolor = ft.Colors.BLACK87
        self.padding = 10
        self.expand = True
        self.expand_loose = True

class Message(ft.Row):
    def __init__(self, text, name, user_or_ai="user"):
        super().__init__()
        self.alignment = (
            ft.MainAxisAlignment.START if user_or_ai == "ai" else ft.MainAxisAlignment.END
        )
        self.vertical_alignment = ft.CrossAxisAlignment.START
        self.user_or_ai = user_or_ai
        if user_or_ai == "ai":
            self.controls = [
                ft.CircleAvatar(bgcolor="blue", content=ft.Text(name, color="white")),
                MessageContainer(body=text),
            ]
        else:
            self.controls = [
                MessageContainer(body=text),
                ft.CircleAvatar(bgcolor="green", content=ft.Text(name, color="white")),
            ]

class Chat(ft.ListView):
    def __init__(self):
        super().__init__()
        self.padding = 10
        self.spacing = 10
        self.auto_scroll = True
        self.expand = True  # สำคัญมากให้ขยายเต็มพื้นที่
        self.controls = []

    def add_message(self, name, text, user_or_ai="user"):
        self.controls.append(Message(text, name, user_or_ai))
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
            else:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": i.controls[0].content.controls[0].value},
                    ]
                })
        return messages

def main(page: ft.Page):
    chat = Chat()

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
    chat.add_message("Ai", 
        """
```sh
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
```
""", "ai")


    page.window.width = 393
    page.window.height = 600
    page.window.always_on_top = False

if __name__ == "__main__":
    ft.app(target=main)
