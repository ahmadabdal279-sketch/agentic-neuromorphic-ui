import json
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END

# 1. LOCAL COGNITIVE INTELLIGENCE BASE
# Local system par 50-year ahead mesh network simulate karne ke liye Llama3 use kar rahe hain
llm = Ollama(model="llama3.2", temperature=0.2)

# 2. SYSTEM STATE DEFINITION
# Yeh state system ke saare agents ke darmiyan memory aur data share karti hai
class AgentState(TypedDict):
    goal: str
    plan: List[str]
    current_execution: str
    critic_feedback: str
    iterations: int
    is_finished: bool

# 3. AGENT 1: THE STRATEGIST
# Iska kaam Goal ko analyze karna aur 50-year ahead architecture ke mutabiq plan banana hai
def strategist_agent(state: AgentState):
    print("\n🤖 [Strategist Agent] Planning the goal...")
    prompt = f"""
    You are the Master Strategist of a 50-year advanced AI mesh.
    User Goal: {state['goal']}
    Current Feedback (if any): {state['critic_feedback']}
    
    Break down this goal into a precise, step-by-step technical plan or conceptual architecture.
    Be extremely dense with high-value technical steps. 
    Respond only with the numbered plan.
    """
    response = llm.invoke(prompt)
    state['plan'] = response.split("\n")
    state['iterations'] += 1
    return state

# 4. AGENT 2: THE EXECUTOR
# Iska kaam Strategist ke banaye huay plan ko simulate aur execute karna hai
def executor_agent(state: AgentState):
    print("\n🛠️ [Executor Agent] Simulating and executing the plan...")
    plan_str = "\n".join(state['plan'])
    prompt = f"""
    You are the Execution Engine. 
    Your goal is to execute the following plan:
    {plan_str}
    
    Generate the complete core solution, comprehensive details, or foundational logic framework for this plan.
    Provide the highest density of actionable data.
    """
    response = llm.invoke(prompt)
    state['current_execution'] = response
    return state

# 5. AGENT 3: THE CRITIC (THE VERIFIER)
# Iska kaam Executor ke kaam mein ghaltiyan nikalna aur perfection check karna hai
def critic_agent(state: AgentState):
    print("\n🕵️ [Critic Agent] Verifying output quality...")
    prompt = f"""
    You are the Critic and Quality Gatekeeper.
    Original Goal: {state['goal']}
    Current Execution Output: {state['current_execution']}
    
    Analyze the execution output. Is it flawless, comprehensive, and fully achieves the goal?
    If yes, reply with EXACTLY the word 'APPROVED'.
    If no, provide a brief, ruthless critique explaining what needs to be fixed or expanded.
    """
    response = llm.invoke(prompt).strip()
    
    if "APPROVED" in response or state['iterations'] >= 3:
        state['is_finished'] = True
        state['critic_feedback'] = "Approved."
    else:
        state['is_finished'] = False
        state['critic_feedback'] = response
    return state

# 6. ROUTING LOGIC (CONDITIONAL EDGE)
# Yeh function faisla karta hai ke kaam khatam ho gaya ya loop dobara chalana hai
def route_next_node(state: AgentState):
    if state['is_finished']:
        return "end"
    else:
        return "strategist"

# 7. BUILDING THE AGENTIC GRAPH MESH
workflow = StateGraph(AgentState)

# Add processing nodes
workflow.add_node("strategist", strategist_agent)
workflow.add_node("executor", executor_agent)
workflow.add_node("critic", critic_agent)

# Set entry point
workflow.set_entry_point("strategist")

# Define execution links
workflow.add_edge("strategist", "executor")
workflow.add_edge("executor", "critic")

# Add 50-year advanced conditional routing loop
workflow.add_conditional_edges(
    "critic",
    route_next_node,
    {
        "strategist": "strategist",
        "end": END
    }
)

# Compile backend mesh system
backend_mesh = workflow.compile()

# --- 🚀 RUNNING THE SYSTEM COMPONENT ---
if __name__ == "__main__":
    # Apna koi bhi advanced goal yahan dein, system autonomous processing shuru kar dega
    user_goal = "Design a system that uses kinetic human typing pressure to charge a laptop battery locally."
    
    initial_state: AgentState = {
        "goal": user_goal,
        "plan": [],
        "current_execution": "",
        "critic_feedback": "",
        "iterations": 0,
        "is_finished": False
    }
    
    print("⚡ Activating 50-Year Ahead Local AI Agentic Mesh...")
    # Nayi fixed line Python 3.14 ke liye:
    final_output = backend_mesh.invoke(initial_state, config={"recursion_limit": 10})

    print("\n=============================================")
    print("✅ SYSTEM EXECUTION COMPLETE (FINAL OUTPUT)")
    print("=============================================")
    print(final_output['current_execution'])
