import json

data = []
with open('dblpv13.json', 'r') as file:
    for line in file:
        try:
            # Attempt to parse each line
            data.append(json.loads(line))
        except json.JSONDecodeError:
            # Skip lines that cause decoding errors
            print(f"Skipping invalid line: {line}")
            continue

# Check if data has been loaded
print(f"Total valid records loaded: {len(data)}")

