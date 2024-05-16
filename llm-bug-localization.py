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
def get_method_body_signature_by_id(project: str, bug_id: str, method_id: str) -> list[str]:
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
def get_method_body_by_method_signature(project: str, bug_id: str,method_signature: str) -> str:
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
def get_stack_traces(project: str, bug_id: str) -> list:
    """Returns the stack traces of a given bug."""
    bugs_data = utils.json_file_to_dict(paths_dict["bugs_with_stack_traces_details_file_path"])
    stack_traces = bugs_data[project][bug_id]["stack_traces"]
    return stack_traces

@tool
def get_test_ids(project: str, bug_id: str) -> list:
    """Returns all the test ids in the data."""
    project_gzoltar_folder = os.path.join(paths_dict["gzoltar_files_path"], project)
    bug_gzoltar_folder = os.path.join(project_gzoltar_folder, bug_id)
    test_names, test_results = utils.read_tests_file(bug_gzoltar_folder)
    return list(range(len(test_names)))

@tool
def get_test_body_by_id(project: str, bug_id: str, test_id: str) -> list[str]:
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




