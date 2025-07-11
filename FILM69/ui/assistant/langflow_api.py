
import requests
import json
from IPython.display import clear_output,Markdown,display

class LangflowAPI:

    def __init__(
            self,
            url="https://film6912-langflow.hf.space",
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
        
        url=f"{self.url}/api/v1/run/{self.flow_id}?stream={str(stream).lower()}"
        
        with requests.post(url, headers=self.headers, json=payload, stream=stream) as response:
            if response.status_code == 200:
                tools_response=[]
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        event_data = json.loads(line)
                        event_type = event_data.get("event")
                        data = event_data.get("data", {})
                        try:
                            tool_block = data.get("content_blocks", [{}])[0].get("contents", [{}])

                            tools_response=[]
                            for i in tool_block:
                                if i.get("type")=="tool_use":
                                    tools_response.append({
                                        "name": i.get("name"),
                                        "input": i.get("tool_input"),
                                        "output": i.get("output")
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
                            "tool":tools_response,
                            "output":text_response[-1]
                            }
                        
                        yield outputs
            else:
                yield {
                    "status":"❌ HTTP Error",
                    "code": response.status_code,
                    "response":response.text
                }

    def get_messages(self,all=False):
        if all:
            url=f"{self.url}/api/v1/monitor/messages"
        else:
            url=f"{self.url}/api/v1/monitor/messages?session_id={self.session_id}"
        response = requests.request("GET", url, headers=self.headers)

        if response.status_code==200:
            data = response.json()

            output = []

            for i in range(0, len(data) - 1, 2):
                user = data[i]
                ai = data[i + 1]
                
                if user["sender"] == "User" and ai["sender"] == "Machine":
                    tool_info = []

                    for block in ai.get("content_blocks", []):
                        for content in block.get("contents", []):
                            if content["type"] == "tool_use":
                                tool_info.append({
                                    "name": content.get("name"),
                                    "input": content.get("tool_input"),
                                    "output": content.get("output")
                                })
                    if len(tool_info)==0:
                        output.append({
                            "input": user["text"],
                            "output": ai["text"],
                            "session_id":data[i]["session_id"]
                        })
                    else:
                        output.append({
                            "input": user["text"],
                            "tool": tool_info,
                            "output": ai["text"],
                            "session_id":data[i]["session_id"]
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
