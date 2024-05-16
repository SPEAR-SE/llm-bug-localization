import utils
import importlib

importlib.reload(utils)

import json
import sys
import os
import re
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
)

from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from langchain_core.tools import tool

import operator
from typing import Annotated, Sequence

from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
import functools

from my_secrets import OPENAI_API_KEY
from my_secrets import base_path

paths_dict = {
    "gzoltar_files_path": os.path.join(base_path, "llm-bug-localization", "data", "gzoltar_files"),
    "bugs_with_stack_traces_details_file_path": os.path.join(base_path, "llm-bug-localization", "data",
                                                             "bug_reports_with_stack_traces_details.json"),
}

projects_folder = {
    "Cli": "commons-cli",
    "Closure": "closure-compiler",
    "Codec": "commons-codec",
    "Compress": "commons-compress",
    "Csv": "commons-csv",
    "Gson": "gson",
    "JacksonCore": "jackson-core",
    "JacksonDatabind": "jackson-databind",
    "Jsoup": "jsoup",
    "JxPath": "commons-jxpath",
    "Lang": "commons-lang",
    "Math": "commons-math",
    "Mockito": "mockito",
    "Time": "joda-time"
}


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
def get_covered_methods_by_failedTest(project: str, bug_id: str, test_id: str) -> list:
    """Returns the covered methods by a failed test. Obs: Returns an empty list if it is a passing test."""
    project_gzoltar_folder = os.path.join(paths_dict["gzoltar_files_path"], project)
    bug_gzoltar_folder = os.path.join(project_gzoltar_folder, bug_id)
    coverage_data = {"statements_covered_per_test": utils.read_matrix_file(bug_gzoltar_folder),
                     "lines_of_code_obj_list": utils.read_spectra_file(bug_gzoltar_folder)}
    test_names, test_results = utils.read_tests_file(bug_gzoltar_folder)
    coverage_data["test_names"] = test_names
    coverage_data["test_results"] = test_results

    covered_methods = []
    if test_id>= 0 and test_id < len(coverage_data["test_names"]):
        test_result = coverage_data["test_results"][test_id]
        if test_result:  # Passing test
            return None
        for index_s, statement_instance in enumerate(
                coverage_data["statements_covered_per_test"][test_id]):
            if str(statement_instance) == "1":  # 1= covered, 0=not covered
                lines_of_code_obj_list = coverage_data["lines_of_code_obj_list"][index_s]
                method = lines_of_code_obj_list["class_name"] + "#" + lines_of_code_obj_list["method_name"]
                if method not in covered_methods:
                    covered_methods.append(
                        lines_of_code_obj_list["class_name"] + "#" + lines_of_code_obj_list["method_name"])

        return covered_methods
    else:
        return None


@tool
def get_method_body_signature_by_id(method_id: str) -> list[str]:
    """
    Takes method_id as parameter and returns the method_id, method_body and signature.
    """
    project_gzoltar_folder = os.path.join(paths_dict["gzoltar_files_path"], project)
    bug_gzoltar_folder = os.path.join(project_gzoltar_folder, bug_id)
    methods_spectra_data = utils.read_methods_spectra_file(bug_gzoltar_folder)
    bugs_data = utils.json_file_to_dict(paths_dict["bugs_with_stack_traces_details_file_path"])

    identifier = methods_spectra_data[method_id]
    repo_path = os.path.join(base_path, "open_source_repos_being_studied", projects_folder[project])
    commit_hash = bugs_data[project][bug_id]["bug_report_commit_hash"]

    # Parse the identifier
    package_class, member_name = identifier.split('#')
    package_name, class_name = package_class.rsplit('.', 1)

    # Checkout the specified commit
    utils.checkout_commit(repo_path, commit_hash)

    # Construct the file path
    file_path = utils.construct_file_path(repo_path, package_name, class_name)

    # Find the method or constructor and the next member
    member, next_member, signature = utils.find_member_and_next(file_path, class_name, member_name)
    if member:
        source_code = utils.extract_source(file_path, member, next_member)
        return [method_id, signature, source_code]
    else:
        return None

@tool
def get_method_body_by_method_signature(method_signature: str) -> str:
    """
    Takes method_signature as parameter and returns the method_body.
    """
    bugs_data = utils.json_file_to_dict(paths_dict["bugs_with_stack_traces_details_file_path"])

    repo_path = os.path.join(base_path, "open_source_repos_being_studied", projects_folder[project])
    commit_hash = bugs_data[project][bug_id]["bug_report_commit_hash"]

    # Split the identifier at the colon to separate package_class and member_name
    package_class, remainder = method_signature.split(':', 1)

    # Now, split off the method arguments by isolating the member name from its parameters
    member_name, _ = remainder.split('(', 1)

    # Finally, separate the package name from the class name by splitting at the last dot in package_class
    package_name, class_name = package_class.rsplit('.', 1)

    # Checkout the specified commit
    utils.checkout_commit(repo_path, commit_hash)

    # Construct the file path
    file_path = utils.construct_file_path(repo_path, package_name, class_name)

    # Find the method or constructor and the next member
    member, next_member, signature = utils.find_member_and_next(file_path, class_name, member_name)
    if member:
        source_code = utils.extract_source(file_path, member, next_member)
        return source_code
    else:
        return None

@tool
def get_stack_traces() -> list:
    """Returns the stack traces of a given bug."""
    bugs_data = utils.json_file_to_dict(paths_dict["bugs_with_stack_traces_details_file_path"])
    stack_traces = bugs_data[project][bug_id]["stack_traces"]
    return stack_traces

@tool
def get_test_ids() -> list:
    """Returns all the test ids in the data."""
    project_gzoltar_folder = os.path.join(paths_dict["gzoltar_files_path"], project)
    bug_gzoltar_folder = os.path.join(project_gzoltar_folder, bug_id)
    test_names, test_results = utils.read_tests_file(bug_gzoltar_folder)
    return list(range(len(test_names)))

@tool
def get_test_body_by_id(test_id: str) -> list[str]:
    """Returns the test body of the given test id."""
    project_gzoltar_folder = os.path.join(paths_dict["gzoltar_files_path"], project)
    bug_gzoltar_folder = os.path.join(project_gzoltar_folder, bug_id)
    test_names, test_results = utils.read_tests_file(bug_gzoltar_folder)
    if test_id>= 0 and test_id < len(test_names):
        test_name = test_names[test_id]

        bugs_data = utils.json_file_to_dict(paths_dict["bugs_with_stack_traces_details_file_path"])
        repo_path = os.path.join(base_path, "open_source_repos_being_studied", projects_folder[project])
        commit_hash = bugs_data[project][bug_id]["bug_report_commit_hash"]

        # Split the identifier at the colon to separate package_class and member_name
        package_class, member_name = test_name.split('#', 1)

        # Finally, separate the package name from the class name by splitting at the last dot in package_class
        package_name, class_name = package_class.rsplit('.', 1)

        # Checkout the specified commit
        utils.checkout_commit(repo_path, commit_hash)

        # Construct the file path
        file_path = utils.construct_file_path(repo_path, package_name, class_name)

        # Find the method or constructor and the next member
        member, next_member, signature = utils.find_member_and_next(file_path, class_name, member_name)
        if member:
            source_code = utils.extract_source(file_path, member, next_member)
            return source_code
        else:
            return None

def get_test_ids_str():
    project_gzoltar_folder = os.path.join(paths_dict["gzoltar_files_path"], project)
    bug_gzoltar_folder = os.path.join(project_gzoltar_folder, bug_id)
    test_names, test_results = utils.read_tests_file(bug_gzoltar_folder)
    if len(test_names) <0:
        return ""
    if len(test_names) == 1:
        return "0"
    if len(test_names) > 1:
        test_ids_str = f"0 to {len(test_names)-1}"
    return test_ids_str


data = ''

reviewer_tools = [get_method_body_by_method_signature]
tester_tools = [get_stack_traces, get_test_body_by_id, get_test_ids, get_covered_methods_by_failedTest]


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
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=OPENAI_API_KEY)

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


def parse_and_save_json(contents, project, bug_id):
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
        "project": project,
        "bug_id": bug_id,
        "ans": json_objects if json_objects else None,
        "final_full_answer": str(contents)
    }

    # Define the output file path
    file_path = f"./data/output/Lang_gpt3_grace_example/{project}_{bug_id}.json"

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

project = sys.argv[1]
bug_id = sys.argv[2]

json_answers = []

test_ids_string = get_test_ids_str()
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
        # parse_and_save_json(value['messages'][0].content, project, bug_id)
        answers.append(value)
    print("\n---\n")

# print(answers)
# Example usage
# content = s['__end__']['messages'][-1].content
parse_and_save_json(answers[:-1], project, bug_id)




