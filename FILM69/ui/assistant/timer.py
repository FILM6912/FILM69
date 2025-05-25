import flet as ft
import time
import threading
import asyncio

class Timer(ft.Column):
    def __init__(self, fn,run_async=False):
        super().__init__()
        self.i=0
        self.fn=fn
        self.run_async=run_async
        if not self.run_async:
            self.th = threading.Thread(target=self.tick, daemon=True)
        self.active = False

    def did_mount(self):
        self.start()
        if not self.run_async:
            self.th.start()
        else:
            self.page.run_task(self.tick1)

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def tick(self):
        while self.active:
            try:self.fn()
            except:...

    async def tick1(self):
        while self.active:
            try:
                self.fn()
                await asyncio.sleep(0)
            except:...


if __name__=="__main__":
    def main(page: ft.Page):
        
        text=ft.Text("0")
        def fn():
            text.value=str(int(text.value)+1)
            text.update()
            print(text.value)
            time.sleep(1)
        
        page.add(Timer(fn),text)
        page.update()

    ft.app(main)