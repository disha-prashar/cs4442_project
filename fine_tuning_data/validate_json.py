import json

# Read JSON file
with open("output.json", "r") as json_file:
    data = json.load(json_file)

proper = 0
improper = 0

# Validate data
for entry in data:
    if len(entry) != 3:
        improper += 1
    else:
        proper += 1

print("Validation complete. The JSON file has ", proper, " proper entries and ", improper, " improper entries.")
