from core.agent_llm import LLM
from core.tools.calculator import math_tool
from core.tools.dictionary import dictionary_tool
from core.retrieval import Retrieval
class Agent:
    def __init__(self):
        self.agent = LLM()
        self.retrieve = Retrieval()
    
    def agent_logic(self, query):
        max_steps = 4
        steps = 0
        current_state = {
            "query":query,
            "input":"",
            "answer":"",
            "tool":"",
            "context":"",
            "scratchpad":"",
            "end":False
        }
        while current_state["end"]!=True and steps < max_steps:
            steps+=1
            print("Step:", steps)
            llm_output = self.agent.agent_llm(current_state)
            print(llm_output)
            for k, v in llm_output.items():
                if k != "scratchpad":
                    current_state[k] = v
            print("State: ", current_state)
            if current_state["end"]:
                break
            if current_state["tool"] == "calculator":
                try:
                    current_state["context"] = math_tool(current_state["input"])
                except:
                    current_state["context"] = "ERROR"
                
            elif current_state["tool"] == "dictionary":
                try:
                    current_state["context"] = dictionary_tool(current_state["input"])
                except:
                    current_state["context"] = "ERROR"

            elif current_state["tool"] == "retrieval":
                current_state["context"] = self.retrieve.hybrid_search(current_state["input"])

            elif current_state["tool"] == "none":
                current_state["end"] = True
                
            if current_state["context"] != "":
                current_state["scratchpad"] += str(current_state["context"]) + "\n"
            print("After State: ", current_state)
            current_state["context"] = ""

        if steps == max_steps:
            return "Agent stopped. Max number of steps reached."
        return current_state["answer"]

if __name__ == "__main__":
    agent = Agent()
    query = "What does velocity mean, and if velocity is 20 m/s, what is 2 x velocity?"
    result = agent.agent_logic(query)
    print(result)