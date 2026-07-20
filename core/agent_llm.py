from groq import Groq
import json, os
from dotenv import load_dotenv

load_dotenv()
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Have to edit the prompts to make it agentic.
class LLM:
    def __init__(self):
        self.client = client_groq
    
    def agent_llm(self, current_state):
        user_prompt = f"""
            Current state of the agent:

            query: {current_state["query"]}
            input: {current_state["input"]}
            answer: {current_state["answer"]}
            tool: {current_state["tool"]}
            context: {current_state["context"]}
            scratchpad: {current_state["scratchpad"]}
            end: {current_state["end"]}

            Decide the next action in JSON format.
            """
        system_prompt = """
            You are a tool-using controller inside a state machine agent.

            Your job is to decide the next action based ONLY on the current state.

            Rules:
            1. Output ONLY valid JSON. No explanations or extra text.
            2. Choose exactly one tool per step: calculator | dictionary | retrieval | none.
            3. "input" must contain the exact query for the selected tool in this step.
            4. "context" contains ONLY the most recent tool output (it is replaced every step).
            5. "scratchpad" contains prior results and must be used to avoid repeating work.
            6. Stop only when the full task is completely solved.

            Tool behavior:
            - calculator: arithmetic or mathematical expressions
            - dictionary: word or concept definitions
            - retrieval: external knowledge search
            - none: final answer is ready

            Multi-step reasoning:
            - Break tasks into steps if needed.
            - Do NOT repeat a tool call if its result already exists in scratchpad.
            - Do NOT set "end": true until all required sub-results are obtained and combined.

            Anti-loop rule:
            - If the same tool input already appears in scratchpad, do NOT call that tool again.

            Output format:
            {
            "input": string,
            "answer": string,
            "tool": "calculator" | "dictionary" | "retrieval" | "none",
            "end": boolean
            }
            """
        chat_completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role":"system",
                    "content": system_prompt

                },
                {
                    "role":"user",
                    "content":user_prompt
                }
            ],
            temperature=0.1,
            max_completion_tokens=300,
            top_p=0.7,
            stream=False,
            response_format={"type": "json_object"}
            )
        answer = chat_completion.choices[0].message.content
        parsed_answer = json.loads(answer)
        return parsed_answer