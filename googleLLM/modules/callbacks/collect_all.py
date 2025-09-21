from langchain.callbacks.base import BaseCallbackHandler


class CollectAllCallback(BaseCallbackHandler):
    def __init__(self):
        self.events = []

    # LLM 프롬프트(원문)
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.events.append({"type": "llm_start", "prompts": prompts})

    # LLM 응답 텍스트(generations)
    def on_llm_end(self, response, **kwargs):
        gens = [[g.text for g in gen] for gen in response.generations]
        self.events.append({"type": "llm_end", "generations": gens})

    # 툴 시작/종료 (예: python_repl_ast 코드 실행)
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.events.append({
            "type": "tool_start",
            "tool": serialized.get("name"),
            "input": input_str,
        })

    def on_tool_end(self, output, **kwargs):
        self.events.append({
            "type": "tool_end",
            "output": output,
        })

    # 에이전트 고수준 액션/종료
    def on_agent_action(self, action, **kwargs):
        self.events.append({
            "type": "agent_action",
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log,
        })

    def on_agent_finish(self, finish, **kwargs):
        self.events.append({
            "type": "agent_finish",
            "return_values": finish.return_values,
        })