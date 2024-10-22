import os
import json
import argparse

# Function to merge JSON files
def merge_json_files(input_folder, output_file):
    # Initialize an empty list to store the merged data
    merged_data = []

    # Iterate over each file in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_folder, file_name)
            
            # Open and load the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Append the content of the JSON file to the merged_data list
                merged_data.append(data)

    # Save the merged data into a single JSON file
    with open(output_file, 'w', encoding='utf-8') as output_file:
        json.dump(merged_data, output_file, indent=4)

    print(f"Merged JSON files into {output_file.name}")

# Set up argument parsing
def main():
    parser = argparse.ArgumentParser(description='Merge multiple JSON files into a single JSON file.')
    
    parser.add_argument('input_folder', type=str, help='Path to the folder containing JSON files.')
    parser.add_argument('output_file', type=str, help='Path to the output JSON file.')

    args = parser.parse_args()

    # Call the function to merge JSON files
    merge_json_files(args.input_folder, args.output_file)

if __name__ == '__main__':
    main()
