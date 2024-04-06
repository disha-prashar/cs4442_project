import json

def parse_text_to_json(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Initialize variables
    json_data_list = []
    current_entry = []

    for line in lines:
        # Strip whitespace from the line
        line = line.strip()
        
        # Check if the line is empty or consists only of delimiters
        if not line or all(delimiter in line for delimiter in ['<|startoftext|>', '<|obj|>', '<|text|>', '<|endoftext|>']):
            # If the current entry is not empty, process it
            if current_entry:
                # Join the lines of the current entry to form a single string
                entry_string = ' '.join(current_entry)
                # Split the entry string into fields based on the delimiters
                fields = entry_string.split("<|startoftext|> <|obj|> <|text|> <|endoftext|>")
                # Create a JSON object for each entry (with only three fields)
                json_entry = {
                    "field1": fields[0].strip(),
                    "field2": fields[1].strip(),
                    "field3": fields[2].strip()
                }
                json_data_list.append(json_entry)
                # Reset current_entry for the next entry
                current_entry = []
        else:
            # Add the line to the current entry
            current_entry.append(line)

    # Write the JSON data to the output file
    with open(output_file, 'w') as f:
        json.dump(json_data_list, f, indent=4)

if __name__ == "__main__":
    input_file = "npc_dialogue_generation.txt"  # Replace with the path to your input text file
    output_file = "output2.json"  # Replace with the desired path for the output JSON file
    parse_text_to_json(input_file, output_file)
