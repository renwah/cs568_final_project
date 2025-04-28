import csv

# Input and output file paths
input_file = 'prompt_examples_dataset.csv'
output_file = 'processed_prompt_examples_dataset.csv'

# Process the CSV file
def process_csv(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = ['id', 'prompt_example', 'prompt_qual'] + reader.fieldnames  # Add 'id' and 'prompt_example' fields to the output
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        row_id = 1  # Initialize row ID
        
        for row in reader:
            # Create a row for "bad_prompt_example"
            bad_prompt_row = row.copy()
            bad_prompt_row['id'] = row_id
            bad_prompt_row['prompt_qual'] = 'bad'
            bad_prompt_row['prompt_example'] = bad_prompt_row.pop('bad_prompt')
            writer.writerow(bad_prompt_row)
            row_id += 1
            
            # Create a row for "good_prompt_example"
            good_prompt_row = row.copy()
            good_prompt_row['id'] = row_id
            good_prompt_row['prompt_qual'] = 'good'
            good_prompt_row['prompt_example'] = good_prompt_row.pop('good_prompt')
            writer.writerow(good_prompt_row)
            row_id += 1

# Run the processing function
process_csv(input_file, output_file)
print(f"Processed data has been saved to {output_file}")