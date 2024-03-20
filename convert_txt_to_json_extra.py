import json

def detect_delimiters(line):
    # Assuming you have a set of possible delimiters to check
    delimiters = [',', ';', '|', '\t']  # Add more delimiters as needed
    detected_delimiters = []
    for delimiter in delimiters:
        if delimiter in line:
            detected_delimiters.append(delimiter)
    return detected_delimiters

def parse_lines_to_json(lines, delimiters):
    # Parse the lines using the detected delimiters
    data = []
    for line in lines:
        for delimiter in delimiters:
            if delimiter in line:
                data.extend(line.strip().split(delimiter))
    json_data = {f"field{i+1}": value for i, value in enumerate(data)}
    return json_data

def convert_text_to_json(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    json_data_list = []
    current_entry_lines = []
    entry_delimiters = []
    for line in lines:
        # Detect all delimiters for each line
        delimiters = detect_delimiters(line)
        if delimiters:
            # Add detected delimiters to the entry_delimiters list
            for delimiter in delimiters:
                if delimiter not in entry_delimiters:
                    entry_delimiters.append(delimiter)
            # If entry_delimiters contain all possible delimiters, it marks the end of the current entry
            if set(entry_delimiters) == set(delimiters):
                # Parse the collected lines into JSON using the detected delimiters
                json_data = parse_lines_to_json(current_entry_lines, entry_delimiters)
                json_data_list.append(json_data)
                # Reset current_entry_lines and entry_delimiters for the next entry
                current_entry_lines = []
                entry_delimiters = []
        # Continue collecting lines for the current entry
        current_entry_lines.append(line)

    # If there are remaining lines after reading all input lines, parse them as well
    if current_entry_lines:
        json_data = parse_lines_to_json(current_entry_lines, entry_delimiters)
        json_data_list.append(json_data)

    with open(output_file, 'w') as f:
        json.dump(json_data_list, f, indent=4)

if __name__ == "__main__":
    input_file = "input.txt"  # Replace with the path to your input text file
    output_file = "output.json"  # Replace with the desired path for the output JSON file
    convert_text_to_json(input_file, output_file)
