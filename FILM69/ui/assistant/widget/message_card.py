import flet as ft
import random
import time
import threading


class MessageCard:
    def __init__(self, user_name, model_name, input_text):
        self.user_name = user_name
        self.model_name = model_name
        self.input_text = input_text
        self.output_text = ""
        self.last_status = None
        self.status = "Processing"
        self.elapsed_time = 0.0
        
        self.processing = {
        }
        
        self.widget_processing = []
        self.create_components()
        self.create_card()
    
    def update_processing_widgets(self):
        """Update the processing widgets based on current data"""
        self.widget_processing.clear()
        
        index=0
        for key in list(self.processing.keys()):
            # Add key row
            if index!=0:
                self.widget_processing.append(ft.Container(height=1,expand=True,bgcolor="#ffffff"))
            index+=1
            self.widget_processing.append(ft.Row([
                ft.Icon(ft.Icons.KEYBOARD_ARROW_RIGHT, color="#ffffff", size=16),
                ft.Markdown(f"#### {key}", selectable=True),
            ], spacing=10))
            
            # Add value container
            self.widget_processing.append(ft.Container(
                # content=ft.Text(self.processing[key], color="#cccccc", size=12),
                content=ft.Markdown(
                    self.processing[key],
                    selectable=True,
                    extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                    code_theme=ft.MarkdownCodeTheme.MONOKAI,
                    code_style_sheet=ft.MarkdownStyleSheet.block_spacing
                    ),
                padding=ft.padding.only(left=30, top=5),
            ))
    
    def create_components(self):
        """Create all UI components with references"""
        # Status components
        self.status_icon = ft.Container(
            content=ft.ProgressRing(width=16, height=16, stroke_width=2, color="#177bec")
        )
        self.status_text = ft.Text("Processing...", color="#ffffff", size=14)
        
        # Time components
        self.time_text = ft.Text("0.0s", color="#06e72c", size=12, weight=ft.FontWeight.BOLD)
        self.time_container = ft.Container(
            content=self.time_text,
            padding=ft.padding.symmetric(horizontal=8, vertical=4),
            border_radius=4,
        )
        
        # Output text components
        self.output_text_display = ft.Markdown(
            "",
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            code_theme=ft.MarkdownCodeTheme.MONOKAI,
            code_style_sheet=ft.MarkdownStyleSheet.block_spacing
            )
        self.output_container = ft.Container(
            content=self.output_text_display,
            padding=ft.padding.only(top=10),
        )
        
        # Create initial processing widgets
        self.update_processing_widgets()
        
        # Container for detailed view
        self.in_card = ft.Column(
            controls=self.widget_processing,
            spacing=5,
        )
    
    def create_card(self):
        """Create the message card UI"""
        self.expansion_tile = ft.ExpansionTile(
            title=ft.Row([
                self.status_icon,
                self.status_text,
                self.time_container
            ], spacing=8),
            controls=[
                ft.Container(
                    content=self.in_card,
                    padding=ft.padding.all(15),
                )
            ],
            bgcolor="#2a2a2a",
            collapsed_bgcolor="#2a2a2a",
            text_color="#ffffff",
            icon_color="#888888",
        )
        
        self.card = ft.Container(
            content=ft.Column([
                # User section (right aligned)
                ft.Row([
                    ft.Markdown(f"### {self.user_name}", selectable=True,),
                    ft.CircleAvatar(
                        content=ft.Text(self.user_name[0].upper(), color="#ffffff", weight=ft.FontWeight.BOLD),
                        bgcolor="#8b5cf6",
                        radius=17,
                    ),
                ], spacing=10, alignment=ft.MainAxisAlignment.END),
                
                # User message (right aligned)
                ft.Container(
                    content=ft.Markdown(self.input_text, selectable=True,),
                    alignment=ft.alignment.center_right,
                    padding=ft.padding.only(right=10, top=5),
                ),
                
                ft.Container(height=5),
                
                # AI section
                ft.Row([
                    ft.Icon(ft.Icons.SMART_TOY, color="#ffffff", size=30),
                    # ft.Text("AI", color="#ffffff", weight=ft.FontWeight.BOLD, size=14),
                    ft.Markdown(f"### {self.model_name}",selectable=True),
                ], spacing=10),
                
                ft.Container(height=5),
                
                # Expandable result card
                ft.Container(
                    content=self.expansion_tile,
                    border_radius=8,
                    border=ft.border.all(1, "#404040"),
                ),
                
                # Output text below the card
                self.output_container,
                
            ]),
            padding=ft.padding.all(20),
            margin=ft.margin.symmetric(horizontal=10, vertical=5),
        )
    
    def update_status(self, status, elapsed_time, data):
        """Update status and time"""
        self.status = status
        self.elapsed_time = elapsed_time
        
        # Update processing data
        if data:
            self.processing.update(data)
            self.update_processing_widgets()
            self.in_card.controls = self.widget_processing

        # Update status icon and text
        if status == "Processing":
            if self.status != self.last_status:
                self.status_icon.content = ft.ProgressRing(width=16, height=16, stroke_width=2, color="#177bec")
                self.last_status = self.status
                
            self.status_text.value = "Processing..."
            # status_color = "#177bec"
            
        elif status == "Generating":
            if self.status != self.last_status:
                self.status_icon.content = ft.ProgressRing(width=16, height=16, stroke_width=2, color="#1fbd00")
                self.last_status = self.status

            self.status_text.value = "Generating..."
            # status_color = "#1fbd00"
            
        else:  # Finished
            self.status_icon.content = ft.Icon(ft.Icons.CHECK_CIRCLE, color="#22c55e", size=22)
            self.status_text.value = "Finished"
            # status_color = "#22c55e"

        # Update time
        self.time_text.value = f"{elapsed_time:.1f}s"
        # self.time_text.color = status_color
    
    def update_output_text(self, text, is_generating=False):
        """Update output text"""
        self.output_text = text
        cursor = "█" if is_generating else ""
        
        # Update main output text
        self.output_text_display.value = text + cursor
    
    def get_card(self):
        """Get the card container"""
        return self.card

def main(page: ft.Page):
    page.title = "Streaming Chat Interface"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#1a1a1a"
    
    message_card = MessageCard(
        user_name="User",
        model_name="models/learnlm-2.0-flash-experimental",
        input_text="สวัสดีครับ คุณสบายดีไหม?"
    )
    
    responses = [
        "สวัสดีครับ มีอะไรให้ช่วยบ้างไหมครับ",
        "ผมชื่อ AI Assistant ยินดีที่ได้รู้จักครับ",
        "ขอบคุณสำหรับคำถามครับ ผมจะช่วยเหลือคุณให้ดีที่สุด",
        "นั่นเป็นคำถามที่น่าสนใจมากครับ ให้ผมอธิบายให้ฟังนะครับ",
    ]
    
    def stream_response(e):
        def run_streaming():
            start_time = time.time()
            response_text = random.choice(responses)
        
            processing_time = random.uniform(1.0, 2.0)
            while time.time() - start_time < processing_time:
                elapsed = time.time() - start_time
                message_card.update_status("Processing", elapsed, {
                    "Input": message_card.input_text,
                })
                message_card.update_output_text("", False)
                page.update()
                time.sleep(0.1)
            
            message_card.update_status("Generating", time.time() - start_time, {
                "Input": message_card.input_text,
            })
            page.update()
            time.sleep(0.5)
            
            current_text = ""
            for i, char in enumerate(response_text):
                current_text += char
                elapsed = time.time() - start_time
                message_card.update_status("Generating", elapsed, {
                    "Input": message_card.input_text,
                    "Tools":current_text,
                    "Output": current_text
                })
                message_card.update_output_text(current_text, True)
                page.update()
                
                time.sleep(0.01)
            
            # Finished
            final_time = time.time() - start_time
            message_card.update_status("Finished", final_time, {
                "Input": message_card.input_text,
                "Tools":current_text,
                "Output": current_text
            })
            message_card.update_output_text(current_text, False)
            page.update()
        
        # Run in thread to avoid blocking UI
        run_streaming()
    
    start_button = ft.ElevatedButton(
        "Start Streaming Response",
        on_click=stream_response,
        bgcolor="#8b5cf6",
        color="#ffffff",
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
        )
    )
    
    page.add(
        message_card.get_card(),
        ft.Container(
            content=start_button,
            alignment=ft.alignment.center,
            padding=ft.padding.all(20),
        )
    )

if __name__ == "__main__":
    ft.app(target=main)