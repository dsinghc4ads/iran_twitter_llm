import csv

"""
Simple script to grab a preset number of rows from a CSV file.
"""
input_file = "data/clean.csv"
output_file = "data/first-20-rows.csv"
max_rows = 20

with (
    open(input_file, "r", encoding="utf-8") as infile,
    open(output_file, "w", encoding="utf-8", newline="") as outfile,
):
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()

    row_count = 0
    for row in reader:
        if row_count >= max_rows:
            break
        writer.writerow(row)
        row_count += 1

print(f"Extracted {row_count} rows to {output_file}")
