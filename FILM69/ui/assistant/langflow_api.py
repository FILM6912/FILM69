
import requests
import json
from IPython.display import clear_output,Markdown,display

class LangflowAPI:

    def __init__(
            self,
            url="http://localhost:7860",
            flow_id="31cd1f8e-a2f3-47f9-8742-773fd1324d79",
            session_id="123",
            api_key=None
            ):
        
        self.session_id=session_id
        self.flow_id=flow_id
        self.url=url
        
        self.headers = {
            'x-api-key':  api_key,
            'Content-Type': 'application/json'
            }

    def chat(self,text_input,stream=True):
        payload = {
            "input_value": text_input,
            "output_type": "chat",
            "input_type": "chat",
            "session_id": self.session_id
            }
        text_response = [""]
        tools_response =[{"name": None,"input": None,"output": None}]
        
        url=f"{self.url}/api/v1/run/{self.flow_id}?stream={str(stream).lower()}"
        
        with requests.post(url, headers=self.headers, json=payload, stream=stream) as response:
            if response.status_code == 200:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        event_data = json.loads(line)
                        event_type = event_data.get("event")
                        data = event_data.get("data", {})
                        try:
                            tool_block = data.get("content_blocks", [{}])[0].get("contents", [{}])[1]
                            tools_response.append({
                                "name": tool_block.get("name"),
                                "input": tool_block.get("tool_input"),
                                "output": tool_block.get("output")
                            })
                        except:...
                        if event_type == "add_message" and data.get("sender") in ["Machine", "AI"]:
                            text = data.get("text", "")
                            if text:
                                text_response.append(text)
                        
                        elif event_type == "end":
                            break
                            
                        outputs={
                            "input":payload["input_value"],
                            "tool":tools_response[-1],
                            "output":text_response[-1]
                            }
                        
                        yield outputs
            else:
                yield {
                    "status":"❌ HTTP Error",
                    "code": response.status_code,
                    "response":response.text
                }
 
    
    def get_messages(self):
        url=f"{self.url}/api/v1/monitor/messages?session_id={self.session_id}"
        response = requests.request("GET", url, headers=self.headers)

        if response.status_code==200:
            data = response.json()

            output = []

            for i in range(0, len(data) - 1, 2):
                user = data[i]
                ai = data[i + 1]

                if user["sender"] == "User" and ai["sender"] == "Machine":
                    tool_info = {"name": None, "input": None, "output": None}

                    for block in ai.get("content_blocks", []):
                        for content in block.get("contents", []):
                            if content["type"] == "tool_use":
                                tool_info = {
                                    "name": content.get("name"),
                                    "input": content.get("tool_input"),
                                    "output": content.get("output")
                                }
                    if tool_info["name"]==None:
                        output.append({
                            "input": user["text"],
                            "output": ai["text"]
                        })
                    else:
                        output.append({
                            "input": user["text"],
                            "tool": tool_info,
                            "output": ai["text"]
                        })

            return output
        else:
                return {
                    "status":"❌ HTTP Error",
                    "code": response.status_code,
                    "response":response.text
                }

    def delete(self,session_id):

        url = f"{self.url}/api/v1/monitor/messages/session/{session_id}"

        response = requests.request("DELETE", url, headers=self.headers)

        return {
            "status":"❌ HTTP Error",
            "code": response.status_code,
            "response":response.text
        }

