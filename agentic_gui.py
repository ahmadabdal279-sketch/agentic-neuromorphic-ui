import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END

# --- AGENTIC BACKEND LOGIC CORE ---
llm = Ollama(model="llama3.2", temperature=0.2)

class AgentState(TypedDict):
    goal: str
    plan: List[str]
    current_execution: str
    critic_feedback: str
    iterations: int
    is_finished: bool

def strategist_agent(state: AgentState):
    update_status("🤖 [Strategist] Designing conceptual framework...")
    prompt = f"You are a Master Strategist AI. User Goal: {state['goal']}\nFeedback: {state['critic_feedback']}\nRespond only with a dense, step-by-step numbered plan."
    response = llm.invoke(prompt)
    state['plan'] = response.split("\n")
    state['iterations'] += 1
    return state

def executor_agent(state: AgentState):
    update_status("🛠️ [Executor] Generating detailed execution logic...")
    plan_str = "\n".join(state['plan'])
    prompt = f"Execute this plan comprehensively with actionable technical details:\n{plan_str}"
    response = llm.invoke(prompt)
    state['current_execution'] = response
    return state

def critic_agent(state: AgentState):
    update_status("🕵️ [Critic] Verifying system output quality...")
    prompt = f"Analyze this output. If it perfectly achieves the goal, reply exactly with 'APPROVED'. Otherwise, give a brief technical critique:\n{state['current_execution']}"
    response = llm.invoke(prompt).strip()
    
    if "APPROVED" in response or state['iterations'] >= 2:
        state['is_finished'] = True
        state['critic_feedback'] = "Approved."
    else:
        state['is_finished'] = False
        state['critic_feedback'] = response
    return state

def route_next_node(state: AgentState):
    return "end" if state['is_finished'] else "strategist"

workflow = StateGraph(AgentState)
workflow.add_node("strategist", strategist_agent)
workflow.add_node("executor", executor_agent)
workflow.add_node("critic", critic_agent)
workflow.set_entry_point("strategist")
workflow.add_edge("strategist", "executor")
workflow.add_edge("executor", "critic")
workflow.add_conditional_edges("critic", route_next_node, {"strategist": "strategist", "end": END})
backend_mesh = workflow.compile()

# --- TKINTER INTERFACE FRONTEND ---
def update_status(text):
    status_label.config(text=text)
    root.update_idletasks()

def run_mesh_thread():
    user_goal = goal_entry.get().strip()
    if not user_goal:
        messagebox.showwarning("Empty Goal", "Please input a valid system goal first.")
        return
    
    submit_btn.config(state=tk.DISABLED)
    progress_bar.start(10)
    output_text.delete("1.0", tk.END)
    
    initial_state: AgentState = {
        "goal": user_goal, "plan": [], "current_execution": "",
        "critic_feedback": "", "iterations": 0, "is_finished": False
    }
    
    def process():
        try:
            final_output = backend_mesh.invoke(initial_state, config={"recursion_limit": 10})
            output_text.insert(tk.END, final_output['current_execution'])
            update_status("✅ Processing Complete! System Idle.")
        except Exception as e:
            update_status("❌ An Error Occurred.")
            messagebox.showerror("Runtime Error", str(e))
        finally:
            progress_bar.stop()
            submit_btn.config(state=tk.NORMAL)

    threading.Thread(target=process, daemon=True).start()

# Window Styling Setup
root = tk.Tk()
root.title("50-Year Ahead Advanced Agentic Mesh UI")
root.geometry("750x600")
root.configure(bg="#1e1e24")

style = ttk.Style()
style.theme_use("clam")
style.configure("TProgressbar", thickness=8, troughcolor="#2d2d38", background="#00adb5")

# UI Layout Grid
tk.Label(root, text="ENTER YOUR AI GOAL SYSTEM:", font=("Helvetica", 11, "bold"), fg="#00adb5", bg="#1e1e24").pack(pady=(15,2))
goal_entry = tk.Entry(root, font=("Consolas", 11), bg="#2d2d38", fg="#ffffff", insertbackground="white", bd=2, relief=tk.FLAT)
goal_entry.pack(fill=tk.X, padx=25, ipady=8)

submit_btn = tk.Button(root, text="DEPLOY AGENTIC LOOP", font=("Helvetica", 10, "bold"), bg="#00adb5", fg="#ffffff", activebackground="#008c95", activeforeground="white", relief=tk.FLAT, command=run_mesh_thread)
submit_btn.pack(pady=12, ipady=5, ipadx=15)

progress_bar = ttk.Progressbar(root, style="TProgressbar", mode="indeterminate")
progress_bar.pack(fill=tk.X, padx=25, pady=5)

status_label = tk.Label(root, text="🟢 System Ready. Awaiting deployment parameter...", font=("Helvetica", 9, "italic"), fg="#a0a0a5", bg="#1e1e24")
status_label.pack(pady=2)

tk.Label(root, text="CHRONO-MESH OUTPUT CONSOLE:", font=("Helvetica", 10, "bold"), fg="#ffffff", bg="#1e1e24").pack(anchor=tk.W, padx=25, pady=(10,2))
output_text = tk.Text(root, font=("Consolas", 10), bg="#141419", fg="#dcdcdc", wrap=tk.WORD, bd=0, padx=10, pady=10)
output_text.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0,20))

root.mainloop()
