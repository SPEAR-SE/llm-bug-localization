import re
import os
import csv
import sys

sys.path.append('../llm-bug-localization')


def read_matrix_file(file_path):
    statements_covered_per_test = []
    test_passed = []

    with open(os.path.join(file_path, "matrix.txt"), 'r') as f:
        for line in f:
            row = [int(num) for num in line.strip()[:-1].split()]
            sign = line.strip()[-1]
            statements_covered_per_test.append(row)
    return statements_covered_per_test


def read_spectra_file(file_path):
    lines_of_code_obj_list = []
    pattern = r'^(.*?)#(.*?)\((.*?)\):(\d+)$'
    with open(os.path.join(file_path, "spectra.csv"), 'r') as file:
        first_line = True
        for line in file:
            # Skip the first line
            if first_line:
                first_line = False
                continue
            composed_str = line
            match = re.search(pattern, composed_str)
            if match is None:
                print("match not found")
                print(composed_str)
                continue
            class_name = match.group(1)
            method_name = match.group(2)
            method_parameters = match.group(3)
            line_number = int(match.group(4))
            lines_of_code_obj_list.append({
                "class_name": class_name,
                "method_name": method_name,
                "method_parameters": method_parameters,
                "line_number": line_number,
            })
    return lines_of_code_obj_list


def read_tests_file(file_path):
    test_names = []
    test_results = []
    with open(os.path.join(file_path, "SBEST_test_results.csv"), 'r') as file:
        content = file.read().replace('\0', '')
        csv_reader = csv.reader(content.splitlines())
        for row in csv_reader:
            test_name = row[0]
            if row[1] == "True":
                test_result = True
            else:
                test_result = False
            test_names.append(test_name)
            test_results.append(test_result)
    return test_names, test_results

