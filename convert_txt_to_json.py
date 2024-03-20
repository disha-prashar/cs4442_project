import json

def parse_line_to_json(line, delimiters):
    # Parse the line using the detected delimiters
    data = []
    if (len(line.split(delimiters[0])) == 1):
        print(line)
    tmp_start = line.split(delimiters[0])[1]
    startoftext = tmp_start.split(delimiters[1])[0]
    data.append(startoftext)
    tmp_obj = tmp_start.split(delimiters[1])[1]
    obj = tmp_obj.split(delimiters[2])[0]
    data.append(obj)
    temp_txt = tmp_obj.split(delimiters[2])[1]
    txt = temp_txt.split(delimiters[3])[0]
    data.append(txt)
    json_data = {f"field{i+1}": value for i, value in enumerate(data)}
    return json_data

def convert_text_to_json(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    json_data_list = []
    current_entry_lines = []
    entry_delimiters = []
    for line in lines:
        delimiters = ['<|startoftext|>', '<|obj|>', '<|text|>', '<|endoftext|>']
        # Parse the line into JSON using the detected delimiters
        json_data = parse_line_to_json(line, delimiters)
        json_data_list.append(json_data)

    with open(output_file, 'w') as f:
        json.dump(json_data_list, f, indent=4)

if __name__ == "__main__":
    input_file = "npc_dialogue_generation.txt" # input text file      npc_dialogue_generation
    output_file = "output.json"  # output JSON file
    convert_text_to_json(input_file, output_file)
