import csv

# Define the input and output filenames
input_filename = "mymoviedb.csv"
output_filename = "mymoviedb_pipe.csv"

# Open the input CSV file for reading and output file for writing
with open(input_filename, mode="r", encoding="utf-8", newline="") as infile, \
     open(output_filename, mode="w", encoding="utf-8", newline="") as outfile:
    
    # Create a CSV reader that handles commas and quoted fields
    reader = csv.reader(infile, delimiter=',', quotechar='"')
    
    # Create a CSV writer that uses pipe as a delimiter
    writer = csv.writer(outfile, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    # Process each row from the input and write it using the new delimiter
    for row in reader:
        writer.writerow(row)

print(f"Conversion complete. Pipe-delimited file saved as '{output_filename}'.")
