import re
import os
import csv
import sys
import subprocess
import javalang
import json
import glob

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


def read_methods_spectra_file(file_path):
    lines_of_code_obj_list = []
    with open(os.path.join(file_path, "methods_spectra.csv"), 'r') as file:
        first_line = True
        for line in file:
            # Skip the first line
            if first_line:
                first_line = False
                continue
            lines_of_code_obj_list.append(line.replace("\n", ""))
    return lines_of_code_obj_list


def checkout_commit(repo_path, commit_hash):
    """Check out a specific commit in a Git repository."""
    os.chdir(repo_path)
    subprocess.run(['git', 'checkout', commit_hash], check=True)


def find_member_and_next(file_path, class_name, member_name):
    """Find the specified method or constructor and the next member in the file."""
    with open(file_path, 'r') as file:
        code = file.read()

    tree = javalang.parse.parse(code)
    # Extract the package name
    package_name = tree.package.name if tree.package else ""
    full_class_name = f"{package_name}.{class_name}" if package_name else class_name

    previous_member = None
    for _, class_node in tree.filter(javalang.tree.ClassDeclaration):
        if class_node.name == class_name:
            members = class_node.constructors + class_node.methods
            for index, member in enumerate(members):
                if member.name == member_name:
                    param_list = ', '.join(f"{param.type.name} {param.name}" for param in member.parameters)
                    signature = ''
                    if isinstance(member, javalang.tree.MethodDeclaration):
                        return_type = member.return_type.name if member.return_type else 'void'
                        signature = f"{full_class_name}:{member_name}({param_list})"
                    elif isinstance(member, javalang.tree.ConstructorDeclaration):
                        signature = f"{full_class_name}:{member_name}({param_list})"

                    # Try to get the next member if possible
                    next_member = members[index + 1] if index + 1 < len(members) else None
                    return member, next_member, signature
                previous_member = member
    return None, None, None


def extract_source(file_path, member, next_member):
    """Extract source code from the file based on member positions and remove multiline comments."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if not hasattr(member, 'position'):
        return "No position data available for this member."

    start_line = member.position.line - 1  # Adjust for 0-based index
    if next_member and hasattr(next_member, 'position'):
        end_line = next_member.position.line - 2  # End before the next member starts
    else:
        end_line = start_line
        while end_line < len(lines) and not lines[end_line].strip().endswith("}"):
            end_line += 1

    # Join lines and remove multiline comments
    code_block = ''.join(lines[start_line:end_line + 1])
    cleaned_code = remove_multiline_comments(code_block)
    return cleaned_code


def remove_multiline_comments(code):
    """Remove multiline comments from the given string of code."""
    pattern = r'/\*.*?\*/'  # Non-greedy match to find /* ... */
    cleaned_code = re.sub(pattern, '', code, flags=re.DOTALL)  # DOTALL to match across lines
    return cleaned_code


def construct_file_path(base_path, package_name, class_name):
    """Construct the file path for a Java class based on package and class name, allowing for flexible filename matching."""
    package_path = package_name.replace('.', '/')
    filename_pattern = f"{class_name}.java"
    possible_base_dirs = ['src', 'src/java', 'src/test']
    print(base_path)
    print(package_name)
    print(class_name)

    for base_dir in possible_base_dirs:
        search_path = os.path.join(base_path, base_dir, package_path, '**', filename_pattern)

        matching_files = glob.glob(search_path, recursive=True)

        if matching_files:
            return matching_files[0]

    return None


def json_file_to_dict(file_path):
    data = {}
    with open(os.path.join(file_path), 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    fp.close()
    return data
