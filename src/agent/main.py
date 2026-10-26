import sys
import os

from typing import TypedDict
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

from langgraph.graph import END, START, StateGraph

MODEL_NAME = os.environ.get("MODEL")
if MODEL_NAME is None:
    raise ValueError("MODEL env var must be present")

BASE_URL="http://localhost:8888/v1"

INITIAL_PROMPT = \
"""
You are an agentic worker. Complete the task at hand.
"""

WORKER_PROMPT = \
"""
You are a worker in a worker-judge loop. Worker tries to complete the task at hand, then Judge evaluates the work state end gives feedback.

The prompt has those key elements:
    - Task — the task being worked on
    - Questions — questions from the Worker to the Judge in previous iteration
    - Feedback — Judge feedback from previous iteration

Examine them all. Then work according to Feedback to get Task closer to being accomplished. DO NOT do anything beyond what is needed to satisfy Feedback.
"""

JUDGE_PROMPT = \
"""
You are a Judge agent in a worker-judge loop. Worker tries to complete the task at hand, then Judge evaluates the work state end gives feedback.

The prompt has those key elements:
    - Task — the task being worked on
    - Questions — questions from Worker to you

Verify the work state thoroughly and decide whether Task has been completed. If it has been completed, output appropriate values for passes and feedback.

If the work doesn't pass, return appropriate feedback and an appropriate value for passes. Make sure feedback answers Questions.

IMPORTANT; you do NOT have to find every mistake or missing work. Once you find a single thing to fix/improve/do, you can immediately proceed to return feedback.
"""


@tool
def system_command(command: str) -> str:
    """Run any system command"""
    return os.system(command)

tools=[system_command]



model = init_chat_model(
        model=MODEL_NAME,
        model_provider="openai",
        base_url=BASE_URL,
        api_key="unnecessary");


class State(TypedDict):
    original_prompt: str 
    questions: str | None
    feedback: str | None
    passes: bool | None

class WorkerResponse(BaseModel):
    questions: str | None = Field(description="Worker questions to judge. Optional")

class JudgeResponse(BaseModel):
    feedback: str | None = Field(description="Judge feedback. None if work is accepted")
    passes: bool = Field(description="Judge verdict for current work state. True means pass; no more work needed. False means changes need to be made based on judge's feedback")

initial = create_agent(
        model=model,
        tools=tools,
        system_prompt=INITIAL_PROMPT,
        response_format=WorkerResponse);

worker = create_agent(
        model=model,
        tools=tools,
        system_prompt=WORKER_PROMPT,
        response_format=WorkerResponse);

judge = create_agent(
        model=model,
        tools=tools,
        system_prompt=JUDGE_PROMPT,
        response_format=JudgeResponse);

def invoke_initial(state):
    content = "# Task:\n"
    content += state["original_prompt"]
    content += "\n\n"
    output = initial.invoke(
            {"messages": [{"role": "user", "content": content}]})
    return {
            "questions": output.questions,
            "feedback": None}

def invoke_worker(state):
    content = "# Task:\n"
    content += state["original_prompt"]
    content += "\n\n"
    content += "# Questions from worker:\n"
    if state["questions"] is not None:
        content += state["questions"]
    else:
        content += "No questions."
    content += "\n\n"
    content += "# Feedback:\n"
    content += state["feedback"]

    output = worker.invoke(
            {"messages": [{"role": "user", "content": content}]})
    return {
            "questions": output.questions,
            "feedback": None}

def check_passes(state):
    return state["passes"]


def invoke_judge(state):
    content = "# Task:\n"
    content += state["original_prompt"]
    content += "\n\n"
    if state["questions"] is not None:
        content += "# Questions:\n"
        content += state["questions"]
        content += "\n\n"

    output = judge.invoke(
            {"messages": [{"role": "user", "content": content}]})
    if output.passes:
        return {
                "feedback": None,
                "passes": True}
    else:
        return {
                "feedback": output.feedback,
                "passes": False}

workflow = StateGraph(State)

workflow.add_node("initial", invoke_initial)
workflow.add_node("worker", invoke_worker)
workflow.add_node("judge", invoke_judge)

workflow.add_edge(START, "initial")
workflow.add_edge("initial", "judge")
workflow.add_conditional_edges(
        "judge", check_passes, {True: END, False: "worker"})
workflow.add_edge("worker","judge")

chain = workflow.compile()

prompt = " ".join(sys.argv)
print(prompt, "++++++++++++++++++")
if prompt is None:
    raise ValueError("No prompt")

def main():
    state = chain.invoke({
        "original_prompt": prompt, 
        "questions": None,
        "feedback": None,
        "passes": False})
    print(state)



