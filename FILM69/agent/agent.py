
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
import nest_asyncio
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

nest_asyncio.apply()
memory = MemorySaver()



@tool()
def list_sum(x=list[int]):
    "รวมค่าใน list"
    return sum(x)

class State(TypedDict):
    messages: Annotated[list, add_messages]

class Agent:
    def __init__(self,model=None,tools_server:list[dict[str]]=None,memory:bool=True,tools:list=None):
        'model=None,tools_server:list[dict[str]]=[{"name":"sse","url":"http://localhost:8000/sse"}],memory:bool=True,tools:list=None'
        
        self.mcp_client = MultiServerMCPClient()
        self.app = asyncio.run(self.make_graph(model,memory,tools_server,tools))
        
    async def make_graph(self,model,used_memory,tools_server,tools):
        
        
        if tools and tools_server:
            for i in tools_server:
                await self.mcp_client.connect_to_server_via_sse(i["name"], url=i["url"])
            
            mcp_tools = self.mcp_client.get_tools()
            llm_with_tools = model.bind_tools(mcp_tools+tools)
            
        elif tools:
            llm_with_tools = model.bind_tools(tools)
        elif tools_server:
            llm_with_tools = model.bind_tools(tools_server)
        else:
            llm_with_tools = model.bind_tools([list_sum])
        
        graph_builder = StateGraph(State)
        
        async def chatbot(state: State) -> State:
            messages = state["messages"]
            response = await llm_with_tools.ainvoke(messages)
            return {"messages": [response]}
        
        graph_builder.add_node("chatbot", chatbot)
        
        graph_builder.add_node('tools', ToolNode(mcp_tools))
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.set_entry_point("chatbot")

        if used_memory:
            app = graph_builder.compile(checkpointer=memory)
        else:
            app = graph_builder.compile()
            
        return app
    
    async def _gen(self,user_input):
        async for event in self.app.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                {"configurable": {"thread_id": "1"}},
            ):
                yield event
                # print(event[list(event.keys())[0]]["messages"][-1].pretty_print())

    def stream(self,text):
        async def _run():
            async for chunk in self.app.astream(
                {"messages": [{"role": "user", "content": text}]},
                {"configurable": {"thread_id": "1"}},
            ):
                yield chunk

        # ตัวช่วยแปลง async generator -> sync generator
        loop = asyncio.get_event_loop()
        gen = _run().__aiter__()

        async def get_next():
            try:
                return await gen.__anext__()
            except StopAsyncIteration:
                return None

        while True:
            result = loop.run_until_complete(get_next())
            if result is None:
                break
            yield result
            
    def invoke(self,user_input):
        events=[]
        for event in self.stream(user_input):
            events.append(event)
        return events
    

if __name__ == "__main__":
    llm=ChatOpenAI(
        base_url="http://127.0.0.1:8080/v1/",
        model="qwen3-4b",
        api_key=""
    )
    tools_server=[{"name":"sse","url":"http://localhost:8000/sse"}]
    
    
    x=Agent(model=llm,memory=False,tools=tools_server)
    for event in x.stream("1+66="):
        print(event[list(event.keys())[0]]["messages"][-1].pretty_print())
        
    print(x.invoke("1+66="))

    

