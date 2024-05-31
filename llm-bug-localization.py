import math

import utils
import glob
import importlib

importlib.reload(utils)

# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain_community.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
import json
import sys
from langchain.agents import AgentType, initialize_agent, load_tools
import os
# from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import AIMessage, HumanMessage
import pdb
from langchain.agents.agent_toolkits.conversational_retrieval.tool import (
    create_retriever_tool,
)
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.agents import AgentActionMessageLog, AgentFinish
import re

from my_secrets import OPENAI_API_KEY
from my_secrets import base_path

MEMORY_KEY = "chat_history"

chat_history = []

paths_dict = {
    "gzoltar_files_path": os.path.join(base_path, "llm-bug-localization", "data", "gzoltar_files"),
    "bugs_with_stack_traces_details_file_path": os.path.join(base_path, "llm-bug-localization", "data",
                                                             "bug_reports_with_stack_traces_details.json"),
    "output_path": os.path.join(base_path, "llm-bug-localization", "data", "output", "langchain_gpt3"),
    "bug_reports_textual_info_path": os.path.join(base_path, "llm-bug-localization", "data", "bug_reports_textual_info"),
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


@tool
def get_methods_covered_by_a_test(test_id: str, page=1) -> (list, int):
    """Returns the covered methods by a test and the number of pages left. It is paginated, so if not page number is informed, it returns the first page"""
    project_gzoltar_folder = os.path.join(paths_dict["gzoltar_files_path"], project)
    bug_gzoltar_folder = os.path.join(project_gzoltar_folder, bug_id)
    coverage_data = {"statements_covered_per_test": utils.read_matrix_file(bug_gzoltar_folder),
                     "lines_of_code_obj_list": utils.read_spectra_file(bug_gzoltar_folder)}
    test_names, test_results = utils.read_tests_file(bug_gzoltar_folder)
    coverage_data["test_names"] = test_names
    coverage_data["test_results"] = test_results

    covered_methods = []
    result = []
    try:
        test_id = int(test_id)
    except ValueError:
        return None
    if test_id >= 0 and test_id < len(coverage_data["test_names"]):
        test_result = coverage_data["test_results"][test_id]
        #if test_result:  # Passing test
        #    return None
        for index_s, statement_instance in enumerate(
                coverage_data["statements_covered_per_test"][test_id]):
            if str(statement_instance) == "1":  # 1= covered, 0=not covered
                lines_of_code_obj_list = coverage_data["lines_of_code_obj_list"][index_s]
                method = lines_of_code_obj_list["class_name"] + "#" + lines_of_code_obj_list["method_name"]
                if method not in covered_methods and "<clinit>" not in method:
                    covered_methods.append(
                        lines_of_code_obj_list["class_name"] + "#" + lines_of_code_obj_list["method_name"])
        methods_per_page = 100
        n_pages = math.ceil(len(covered_methods) / methods_per_page)
        if n_pages > 1:
            start = (page - 1) * methods_per_page
            end = page * methods_per_page
            if end > len(covered_methods):
                end = len(covered_methods)
            result = covered_methods[start:end]
        else:
            result = covered_methods

        return (result, n_pages)
    else:
        return (None, 0)


@tool
def get_tests_that_better_cover_the_stack_trace() -> list:
    """Returns a list of test ids from the 5 tests that better cover the stack trace."""
    project_gzoltar_folder = os.path.join(paths_dict["gzoltar_files_path"], project)
    bug_gzoltar_folder = os.path.join(project_gzoltar_folder, bug_id)
    coverage_data = {"statements_covered_per_test": utils.read_matrix_file(bug_gzoltar_folder),
                     "lines_of_code_obj_list": utils.read_spectra_file(bug_gzoltar_folder)}
    test_names, test_results = utils.read_tests_file(bug_gzoltar_folder)
    coverage_data["test_names"] = test_names
    coverage_data["test_results"] = test_results

    tests_list = []
    for test_id in range(len(coverage_data["test_names"])):
        test_result = coverage_data["test_results"][test_id]
        if not test_result:  # Fake failing test
            tests_list.append(test_id)
    return tests_list


@tool
def get_bug_report_textual_info():
    """
    Returns the textual info contained the bug report (title and description)
    """
    txt_file = os.path.join(paths_dict["bug_reports_textual_info_path"], f"{project}_{bug_id}.txt")
    file_content = ""
    if os.path.isfile(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as file_obj:
            file_content = file_obj.read()
    return file_content


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
    print(method_signature)
    if "(" in method_signature:
        print("1!")
        method_signature = method_signature.split("(")[0]
    if ":" in method_signature:
        print("2!")
        package_class, remainder = method_signature.split(':', 1)

        # Now, split off the method arguments by isolating the member name from its parameters
        member_name, _ = remainder.split('(', 1)

        # Finally, separate the package name from the class name by splitting at the last dot in package_class
        package_name, class_name = package_class.rsplit('.', 1)
    elif "#" in method_signature and method_signature.count('$') == 3:
        print("3!")
        m_signature, remainder = method_signature.split('"#"', 1)
        package_class, member_name = method_signature.split('$', 1)
        # Finally, separate the package name from the class name by splitting at the last dot in package_class
        package_name, class_name = package_class.rsplit('$', 1)
    elif "#" in method_signature and method_signature.count('$') == 2:
        print("3.5!")
        method_signature = method_signature.split("#")[0]
        package_class, member_name = method_signature.split('$', 1)
        # Finally, separate the package name from the class name by splitting at the last dot in package_class
        package_name, class_name = package_class.rsplit('.', 1)
    elif "#" in method_signature and '$' in method_signature:
        print("4!")
        package_class, member_name = method_signature.split('#', 1)
        # Finally, separate the package name from the class name by splitting at the last dot in package_class
        #if project=="Jsoup":
        #    package_name, class_name = package_class.rsplit('.', 1)
        #else:
        package_name, class_name = package_class.rsplit('$', 1)
    elif "#" in method_signature:
        print("5!")
        package_class, member_name = method_signature.split('#', 1)
        # Finally, separate the package name from the class name by splitting at the last dot in package_class
        package_name, class_name = package_class.rsplit('.', 1)
    elif "$" in method_signature:
        if project == "Csv" or project == "JaksonCore":
            print("5.5!")
            package_class, member_name = method_signature.rsplit('.', 1)
            package_name, class_name = package_class.rsplit('$', 1)
        else:
            print("6!")
            package_class, member_name = method_signature.rsplit('.', 1)
            package_name, class_name = package_class.rsplit('.', 1)
    else:
        print("7!")
        parts = method_signature.split('.')
        if len(parts) > 2:
            print("8!")
            package_class, member_name = method_signature.rsplit('.', 1)
            package_name, class_name = package_class.rsplit('.', 1)
        else:
            print("9!")
            class_name, member_name = method_signature.split('.', 1)
            package_name = ""

    # Checkout the specified commit
    utils.checkout_commit(repo_path, commit_hash)

    # Construct the file path
    print(repo_path)
    print(package_name)
    print(class_name)
    c_name = class_name
    if '$' in c_name:
        c_name, class_name = class_name.rsplit('$', 1)
    file_path = utils.construct_file_path(repo_path, package_name, c_name)

    # Find the method or constructor and the next member
    if (
            member_name == "invokeNative" or member_name == "InvocationTargetException") and project == "Gson" and bug_id == "8":
        return None
    if member_name == "testFails" and project == "JacksonDatabind" and bug_id == "59":
        return None
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
    try:
        test_id = int(test_id)
    except ValueError:
        return None
    if test_id >= 0 and test_id < len(test_names):
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
    if len(test_names) < 0:
        return ""
    if len(test_names) == 1:
        return "0"
    if len(test_names) > 1:
        test_ids_str = f"0 to {len(test_names) - 1}"
    return test_ids_str


tools = [get_method_body_by_method_signature, get_stack_traces, get_test_body_by_id, get_test_ids,
         get_methods_covered_by_a_test, get_tests_that_better_cover_the_stack_trace, get_bug_report_textual_info]

project = sys.argv[1]
bug_id = sys.argv[2]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
#output_file_path = os.path.join(paths_dict["output_path"], f"{project}_{bug_id}.json")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a Tester agent. You will be presented with an stack trace, and tools (functions) to access the source code and (passing) tests information of the system. Your task is list all the suspicious methods which needs to be analyzed to find the fault. You will be given 4 chances to interact with functions to gather relevant information.
                """,
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        # | condense_prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()

    # | output_parser

)

user_input = """
As a Tester agent, I want you to list all the methods which might be suspicious to find the fault in the system under test. First analyze the bug report and stack trace and then based on that look for the covered methods by the important tests which might be suspicious or leading to the fault.

You Must conclude your analysis with a JSON object ranking the top 5 suspicious  methods and summarizing your reasoning, following the specified structure: 
```json
{
        "method_signatures": [sig1, sig2, sig3, sig4, sig5]  // The potential suspicious method's signatures
}
```
"""

intermediate_steps = []

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
candidates_path = os.path.join(paths_dict["output_path"], "candidates", f"{project}_{bug_id}.json")
raw_output_path = os.path.join(paths_dict["output_path"], "raw_output", f"{project}_{bug_id}.json")
utils.save_raw_output(result['output'], raw_output_path)

utils.parse_and_save_methodsig_json_2(result['output'], project, bug_id, candidates_path)
