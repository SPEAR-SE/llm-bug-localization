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
}



#@tool
def get_covered_methods_by_failedTest(project: str, bug_id: str, test_id: str) -> list:
    """Returns the covered methods by a failed test. Obs: Returns an empty list if it is a passing test."""
    project_gzoltar_folder = os.path.join(paths_dict["gzoltar_files_path"], project)
    bug_gzoltar_folder = os.path.join(project_gzoltar_folder, bug_id)
    coverage_data = {}
    coverage_data["statements_covered_per_test"] = utils.read_matrix_file(bug_gzoltar_folder)
    coverage_data["lines_of_code_obj_list"] = utils.read_spectra_file(bug_gzoltar_folder)
    test_names, test_results = utils.read_tests_file(bug_gzoltar_folder)
    coverage_data["test_names"] = test_names
    coverage_data["test_results"] = test_results

    covered_methods = []
    for index_t, test_coverage in enumerate(coverage_data["statements_covered_per_test"]):
        test_name = coverage_data["test_names"][index_t]
        test_result = coverage_data["test_results"][index_t]
        if test_name == test_id:
            if test_result:  # Passing test
                return []
            for index_s, statement_instance in enumerate(
                    coverage_data["statements_covered_per_test"][index_t]):
                if str(statement_instance) == "1":  # 1= covered, 0=not covered
                    lines_of_code_obj_list = coverage_data["lines_of_code_obj_list"][index_s]
                    method = lines_of_code_obj_list["class_name"] + "#" + lines_of_code_obj_list["method_name"]
                    if method not in covered_methods:
                        covered_methods.append(lines_of_code_obj_list["class_name"] + "#" + lines_of_code_obj_list["method_name"])
    return covered_methods



print(
    get_covered_methods_by_failedTest("Cli", "5", "org.apache.commons.cli.LongOptionWithShort#testLongOptionWithShort"))
