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
    tests_file = os.path.join(file_path, "SBEST_test_results.csv")
    if not os.path.exists(tests_file):
        tests_file = os.path.join(file_path, "test_results_original_ochiai.csv")
    with open(tests_file, 'r') as file:
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
    print(file_path)
    print(class_name)
    print(member_name)
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
    possible_base_dirs = ['src', 'src/java', 'src/test', 'src/main/java', 'gson/src/main/java', 'src/test/java',
                          'test/org']

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


def save_raw_output(output, file_path):
    # Create the directory if it doesn't exist
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(output, f, indent=4)


def is_duplicate(new_obj, existing_objects):
    # Check if new_obj is in existing_objects based on 'test_id' and 'method_signatures'
    for obj in existing_objects:
        if obj['test_id'] == new_obj['test_id'] and set(obj['method_signatures']) == set(new_obj['method_signatures']):
            return True
    return False


def parse_and_save_methodsig_json_2(contents, project_name, bug_id, path):
    json_objects = []
    code_blocks = re.findall(r'```json\n([\s\S]*?)\n```', contents)

    for block in code_blocks:
        try:
            json_obj = json.loads(block)
            if not is_duplicate(json_obj, json_objects):
                json_objects.append(json_obj)
        except json.JSONDecodeError:
            continue  # Skip blocks that cannot be parsed as JSON

    # Handle the case where no valid JSON objects are found
    if json_objects:
        # If there's only one object, use it directly; otherwise, use the whole list
        final_json = json_objects[0] if len(json_objects) == 1 else json_objects
    else:
        # Initialize as an empty dict or with default structure when no data is found
        final_json = {
            "project_name": project_name,
            "bug_id": bug_id,
            "method_signatures": [],
            "final_ans": contents
        }

    # Assign additional properties to the final_json
    if isinstance(final_json, dict):
        final_json['project_name'] = project_name
        final_json['bug_id'] = bug_id
        final_json['final_ans'] = contents

    # Define the output file path
    file_path = path

    # Create the directory if it doesn't exist
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    # Write the combined data to the file
    with open(file_path, "w") as json_file:
        json.dump(final_json, json_file, indent=4)

    print(f"Data saved to {file_path}")
    return file_path
