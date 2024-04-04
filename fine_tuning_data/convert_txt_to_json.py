'''
This program was developed to transform the dataset found here (https://jakub.thebias.nl/GPT2_WOWHead_dataset.txt) into a JSON file.
The file output.json was then uploaded to HuggingFace and can be found here (https://huggingface.co/datasets/dprashar/npc_dialogue_rpg_quests)
'''
import json

# Detects delimiters found in current line being parsed
def detect_delimiters(line):
    delimiters = ['<|startoftext|>', '<|obj|>', '<|text|>', '<|endoftext|>']
    detected_delimiters = []
    for delimiter in delimiters:
        if delimiter in line:
            detected_delimiters.append(delimiter)
    return detected_delimiters

# Receives current entry lines array and delimiters
def parse_lines_to_json(lines):
    # List of delimiters found in every line of the text file
    delimiters = ['<|startoftext|>', '<|obj|>', '<|text|>', '<|endoftext|>']

    # Parse the lines using the detected delimiters
    data = []
    for line in lines:
        for delimiter in delimiters:
            if delimiter in line:
                # data.extend(line.strip().split(delimiter))
                if delimiter == '<|startoftext|>' and '<|obj|>' in delimiters:
                    # Parse the line using the detected delimiters
                    tmp_start = line.split(delimiter)[1]
                    startoftext = tmp_start.split('<|obj|>')[0].strip()
                    data.append(startoftext)
                elif delimiter == '<|obj|>' and '<|text|>' in delimiters:
                    tmp_obj = line.split(delimiter)[1]
                    obj = tmp_obj.split('<|text|>')[0].strip()
                    data.append(obj)
                elif delimiter == '<|text|>' and '<|endoftext|>' in delimiters:
                    temp_txt = line.split(delimiter)[1]
                    txt = temp_txt.split('<|endoftext|>')[0].strip()
                    data.append(txt)
    # Build JSON data structure
    titles = ["Title", "Objective", "Text"]
    json_data = {titles[i]: value for i, value in enumerate(data)}
        
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
        # Only processes if delimiters are found
        if delimiters:
            # Add detected delimiters to the entry_delimiters list
            for delimiter in delimiters:
                if delimiter not in entry_delimiters:
                    entry_delimiters.append(delimiter)
            # If entry_delimiters contain all possible delimiters, it marks the end of the current entry
            if (delimiters[0] == '<|endoftext|>'):
                current_entry_lines[-1] += (line.strip())
            if set(entry_delimiters) == set(delimiters):
                # Parse the collected lines into JSON using the detected delimiters
                json_data = parse_lines_to_json(current_entry_lines)
                if (bool(json_data) and len(json_data) == 3):
                    json_data_list.append(json_data)
                # Reset current_entry_lines and entry_delimiters for the next entry
                current_entry_lines = []
                entry_delimiters = []
            # Continue collecting lines for the current entry
            if (delimiters[0] != '<|endoftext|>'):
                current_entry_lines.append(line.strip())
        # Appends current line to previous line found
        else:
            current_entry_lines[-1] += line.strip()
        
    # If there are remaining lines after reading all input lines, parse them as well
    if current_entry_lines:
        json_data = parse_lines_to_json(current_entry_lines)
        if (bool(json_data) and len(json_data) == 3):
            json_data_list.append(json_data)

    # Writes final JSON entries to the JSON file specified by output_file
    with open(output_file, 'w') as f:
        json.dump(json_data_list, f, indent=4)

# Main entry point to program
if __name__ == "__main__":
    input_file = "npc_dialogue_generation.txt" # input text file
    output_file = "output.json"  # output JSON file
    convert_text_to_json(input_file, output_file)
