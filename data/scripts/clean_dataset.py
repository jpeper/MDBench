import json
import argparse

# Function to remove specific fields from each entry in the JSON data
def remove_fields_from_json(input_file, output_file, fields_to_remove):
    # Load the JSON data from the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Iterate over each entry in the JSON data and remove specified fields
    for entry in data:
        for field in fields_to_remove:
            if field in entry:
                del entry[field]

    # Save the modified data back to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

    print(f"Removed fields {fields_to_remove} from each entry and saved to {output_file}")

# Set up argument parsing
def main():
    parser = argparse.ArgumentParser(description='Remove specific fields from each entry in a JSON file.')
    
    parser.add_argument('input_file', type=str, help='Path to the input JSON file.')
    parser.add_argument('output_file', type=str, help='Path to the output JSON file.')
    
    args = parser.parse_args()

    # Fields to remove from each entry
    fields_to_remove = ["validity_scalar", "example_id"]

    # Call the function to remove the fields
    remove_fields_from_json(args.input_file, args.output_file, fields_to_remove)

if __name__ == '__main__':
    main()