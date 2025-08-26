import csv

def sort_csv_by_score(csv_file_path, output_file_path):
    # Read the CSV file
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    # Sort the rows by the "SCORE" column in descending order
    sorted_rows = sorted(rows, key=lambda row: float(row['SCORE']), reverse=True)

    # Write the sorted rows to a new CSV file
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"Sorted CSV has been saved to {output_file_path}")

# Example usage
csv_file_path = 'Articles/PM/Graded_articles/graded_articles2.csv'  # Replace with your input CSV file path
output_file_path = 'sorted_output.csv'  # Replace with your desired output CSV file path
sort_csv_by_score(csv_file_path, output_file_path)