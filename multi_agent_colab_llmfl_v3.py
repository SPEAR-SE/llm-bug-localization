import pdb
import json
import sys
import os
import re
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)

from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults

import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union

from langchain.agents import create_openai_functions_agent
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
import functools


def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    functions = [format_tool_to_openai_function(t) for t in tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Fault localization AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your final response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)

@tool
def get_covered_method_by_failedTest(test_id: int) -> list:
    """Returns the covered methods by a failed test."""
    for i in data['tests']:
        current_test_id = i['test_id']
        if current_test_id == test_id:
            covered_methods = i['covered_methods']
    return covered_methods

@tool
def get_method_body_signature_by_id(method_id: str) -> str:
    """
    Takes method_id as parameter and returns the method_id, method_body and signature.
    """
    for test in data['tests']:
        for method in test['covered_methods']:
            if method['method_id'] == method_id:
                return [method_id, method['method_signature'],method['method_body']]

    return "Method body not found."

@tool
def get_method_body_by_method_signature(method_signature: str) -> str:
    """
    Takes method_signature as parameter and returns the method_body.
    """
    for test in data['tests']:
        for method in test['covered_methods']:
            if method['method_signature'] == method_signature:
                return [method['method_body']]

    return "Method body not found."

@tool
def get_method_body_and_id(test_id: int, method_id: int) -> list:
    """
    Takes test id and methid as parameter. Returns the method body and id based on the test_id and method_id from the given data.
    """
    for test in data['tests']:
        if test['test_id'] == test_id:
            # Iterate through covered methods
            for method in test['covered_methods']:
                if method['method_id'] == method_id:
                    return [method['method_id'],method['method_body']]

    return "Method body not found."


@tool
def get_covered_method_by_failedTest(test_id: int) -> list:
    """Returns the covered methods by a failed test."""
    for i in data['tests']:
        current_test_id = i['test_id']
        if current_test_id == test_id:
            covered_methods = i['covered_methods']
    return covered_methods

@tool
def get_covered_method_ids_by_failedTest(test_id: int) -> list:
    """Returns the method ids covered by a failed test."""
    method_ids = []
    for i in data['tests']:
        current_test_id = i['test_id']
        if current_test_id == test_id:
            for method in i['covered_methods']:
                method_ids.append(method['method_id'])
    return method_ids



@tool
def get_test_name(test_id: int) -> list:
    """Returns the test name for a specific test id."""
    for i in data['tests']:
        current_test_id = i['test_id']
        if current_test_id == test_id:
            test_name = i['test_name']
            return test_name

@tool
def get_test_ids(bug_id: int) -> list:
    """Returns all the test ids in the data."""
    test_ids = []
    if data['bug_id'] == int(bug_id):
        for i in data['tests']:
            current_test_id = i['test_id']
            test_ids.append(current_test_id)
    return test_ids


@tool
def get_test_body_stacktrace(test_id: int) -> list:
    """Returns the test body of the failed test."""
    test_body = ''
    for i in data['tests']:
        current_test_id = i['test_id']
        if current_test_id == test_id:
            test_body = i['test_body']
            stacktreace = i['stack_trace']
    return [test_body, stacktreace]

@tool
def covered_class_variables(test_id: int) -> str:
    """Returns the covered class variables by a failed test. If there are no data ignore this function."""
    for i in data['tests']:
        current_test_id = i['test_id']
        if current_test_id == test_id:
            covered_statements = i['covered_class_variables']
    return covered_statements


@tool
def get_stacktrace(test_id: int) -> str:
    """Returns the stacktrace of the failed test."""
    stakctrace = ''
    for i in data['tests']:
        current_test_id = i['test_id']
        if current_test_id == test_id:
            stakctrace = i['stack_trace']
    return stakctrace

@tool
def get_test_callgraph(test_id: int) -> list:
    """Returns the callgraph of the failed test in a sequence. The returned id is a list of method ids."""
    for i in data['tests']:
        current_test_id = i['test_id']
        if current_test_id == test_id:
            callgraph = i['call_graph']
    return callgraph


data = ''

reviewer_tools = [ get_method_body_by_method_signature]
tester_tools = [get_test_body_stacktrace, get_test_ids, get_covered_method_by_failedTest]



# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str




# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


# llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Research agent and node
tester_agent = create_agent(
    llm,
    tester_tools,
    system_message="You are a tester. You should gather all the test_ids, test_body and stack traces. Communicate with the debugger if you need more information about any method.",
)
tester_node = functools.partial(agent_node, agent=tester_agent, name="Tester")

# Chart Generator
debugger_agent = create_agent(
    llm,
    reviewer_tools,
    system_message="You are a debugger. You should communicate with tester and analyze the method which tester asks you to analyze and send a report to the tester agent.",
)
debugger_node = functools.partial(agent_node, agent=debugger_agent, name="Debugger")



# # llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
# llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# # Research agent and node
# tester_agent = create_agent(
#     llm,
#     tester_tools,
#     system_message="You should gather all the test_ids, test_body and stack traces.",
# )
# research_node = functools.partial(agent_node, agent=research_agent, name="Tester")

# # Chart Generator
# debugger_agent = create_agent(
#     llm,
#     reviewer_tools,
#     system_message="You should gather all the method_ids covered by a test_id and analyze the method bodies covered by a test_id.",
# )
# chart_node = functools.partial(agent_node, agent=chart_agent, name="Debugger")



tools = tester_tools + reviewer_tools
tool_executor = ToolExecutor(tools)


def tool_node(state):
    """This runs tools in the graph

    It takes in an agent action and calls that tool and returns the result."""
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    # We can pass single-arg inputs by value
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}



# Either agent can decide to end
def router(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        # The previus agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "end"
    # if "```json" in last_message.content:
    #     # Any agent decided the work is done
    #     return "end"
    if "successfully completed" in last_message.content:
        # Any agent decided the work is done
        return "end"
    if "Great collaboration!" in last_message.content:
        # Any agent decided the work is done
        return "end"
    if "Great collaboration" in last_message.content:
        # Any agent decid
        return "end"
    return "continue"




def parse_and_save_json(contents, project_name, bug_id):
    # Initialize a list to store JSON objects found
    json_objects = []

    # Loop through each content string in the contents list
    for content in contents:
        # Extract JSON strings from code blocks marked with triple backticks specifically for JSON
        # print(content['messages'][0].content)
        code_blocks = re.findall(r'```json\n([\s\S]*?)\n```', content['messages'][0].content)
        for block in code_blocks:
            try:
                json_obj = json.loads(block)
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                continue  # Skip blocks that cannot be parsed as JSON

        # # If no JSON objects were found in code blocks, search the entire content for standalone JSON objects
        # if not json_objects:
        #     standalone_json_strings = re.findall(r'\{.*?\}', content['messages'][0].content, re.DOTALL)
        #     for json_string in standalone_json_strings:
        #         try:
        #             json_obj = json.loads(json_string)
        #             json_objects.append(json_obj)
        #         except json.JSONDecodeError:
        #             continue  # Skip strings that cannot be parsed as JSON

    # Prepare the final JSON structure
    final_json = {
        "project_name": project_name,
        "bug_id": bug_id,
        "ans": json_objects if json_objects else None,
        "final_full_answer": str(contents)
    }

    # Define the output file path
    file_path = f"/Users/user/Desktop/llmfl/llama-index-test/RandomTesting/data/output/Lang_gpt3_grace_example/{project_name}_{bug_id}.json"

    # Create the directory if it doesn't exist
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    # Write the combined data to the file
    with open(file_path, "w") as json_file:
        json.dump(final_json, json_file, indent=4)
    
    print(f"Data saved to {file_path}")
    return file_path


workflow = StateGraph(AgentState)

workflow.add_node("Tester", tester_node)
workflow.add_node("Debugger", debugger_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Tester",
    router,
    {"continue": "Debugger", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "Debugger",
    router,
    {"continue": "Tester", "call_tool": "call_tool", "end": END},
)

workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "Tester": "Tester",
        "Debugger": "Debugger",
    },
)
workflow.set_entry_point("Tester")
graph = workflow.compile()



from langchain.output_parsers import StructuredOutputParser, ResponseSchema

test_id_schema = ResponseSchema(name="test_id",
                             description="The id of the test")

# test_name_schema = ResponseSchema(name="test_name",
#                              description="The name of the test")

suspicious_method_schema = ResponseSchema(name="method_signature",
                                      description="The most suspicious method's signature")


suspicious_method_reason_schema = ResponseSchema(name="reasoning",
                                      description="The reason for the method being suspicious")


response_schemas = [suspicious_method_schema, suspicious_method_reason_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()



project_name = sys.argv[1]
bug_id = sys.argv[2]

json_answers = []



def parse_json_for_test_ids(json_data):
    # Load the JSON data if it's a string; otherwise assume it's already a dict
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Extract the test_ids, assuming they are integers
    test_ids = [int(test["test_id"]) for test in data.get("tests", [])]
    
    # Find unique test_ids and sort them
    unique_test_ids = sorted(set(test_ids))
    
    # Check if there's only one unique test_id
    if len(unique_test_ids) == 1:
        return str(unique_test_ids[0])
    
    # If there are multiple test_ids, format them into a range string
    if unique_test_ids:
        test_ids_str = f"{unique_test_ids[0]} to {unique_test_ids[-1]}"
    else:
        test_ids_str = ""  # Return an empty string if there are no test_ids
    
    return test_ids_str

# File path to your JSON file
file_path = f'/Users/user/Desktop/llmfl/llama-index-test/data/Lang/processed_by_grace_withoutline/{bug_id}/{project_name}_{bug_id}.json'

# Load JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

test_ids_string = parse_json_for_test_ids(data)
print(f"Test IDs: {test_ids_string}")

# prompt1 = f"""
# As a Debugger, your primary objective is to identify the fault in the methods covered by failing test for bug id {bug_id}. The fault can be fix in a method or any removal of code in a method. You must use the tools to retrive necessary information. Examine test_id {test_ids_string}. Use the test_id {test_ids_string} and call `get_test_body` and `get_stacktrace` to understand the test and where it fails. Then gather all the method ids covered by test_id {test_ids_string} by calling `get_covered_method_ids_by_failedTest`. Then you analyze all the methods covered methods by test_id {test_ids_string}. `get_method_body_signature_by_id` will provide you the method's id and body when given its method_id. Analyze the covered method ids so that you can understand what each method do and how it is interconnected with other methods. Sometimes the method might not be directly related with the test. There can be multiple methods with the same name but different parameters. Your task is to identify the method where is the fault and in which method the fix should happen.
# Carefully analyze the code of each method, looking for how the fault might originate or propagate. Your goal is to identify the most suspicious method. If the agents feel like they have identified the faulty methods then FINISH. 

# Conclude your analysis with a JSON object ranking these methods and summarizing your reasoning in one sentence. The final answer must follow the specified structure: {format_instructions}. If you feel like you have identified the faulty methods then STOP and FINISH. 
# """

# prompt2 = f"""
# Fault Localization Workflow for Bug ID: {bug_id}

# Objective: Collaboratively identify faulty methods causing a specific bug, with the Tester analyzing test details and the Debugger examining methods suspected of faults.

# Step-by-Step Analysis:

# Step 1: Tester Analysis
# - Action: Analyze the test details for `test_id` {test_ids_string} using `get_test_body_stacktrace`. 
# - Outcome: Analyze the test body to know the intent of the test. Then analyze the stacktrace to get insight about why this test might be failing.

# Step 2: Method Selection by tester and direction for the Debugger
# - Action: Gather all methods related to the test failure using `get_covered_method_by_failedTest`. 
#  Compile a list of potentially faulty methods based on the test analysis. 
# - Select and direct the Debugger to investigate all methods suspected of faults by providing their `method_signature`. 

# Step 3: Debugger Method Analysis
# - Action: Retrieve and analyze each suspected method's body with `get_method_body_by_method_signature`. The retrieved method body will contain only the lines covered by the test. 
# - Outcome: Conduct an in-depth analysis of logic, parameters, and interactions. 

# Step 4: Collaborative Review
# - Action: Discuss findings to determine if the method contains the fault.
# - Outcome: If a fault is identified by agreement of tester and debugger, conclude the analysis; otherwise, the tester should ask the debugger to analyze other method. 


# Conclusion and Format for Reporting:
# - Cease analysis upon fault identification or continue as needed based on Tester's guidance and collaborative review. You must provide final findings in JSON format as per {format_instructions}, including the most suspicious method name and reasons.

# Note: Effective communication and feedback between the Tester and Debugger are crucial for success.
# """

# Read the prompt
with open('data/prompts/prompt2.txt', 'r') as file:
    file_contents = file.read()

prompt2 = f"{file_contents.format(bug_id=bug_id, test_ids_string=test_ids_string, format_instructions=format_instructions)}"

answers = []

for s in graph.stream(
    {
        "messages": [
            HumanMessage(
                content=prompt2
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
):
    for key, value in s.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
        # parse_and_save_json(value['messages'][0].content, project_name, bug_id)
        answers.append(value)
    print("\n---\n")



# print(answers)
# Example usage
# content = s['__end__']['messages'][-1].content
parse_and_save_json(answers[:-1], project_name, bug_id)

# print(s)