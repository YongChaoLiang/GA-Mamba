"""
Chemical formula encoding
"""
import re
import csv
def parse_formula(formula):
    """Parse a chemical formula and return a dictionary of element atom counts"""
    elements = {}
    # Enhanced regular expression to handle various edge cases
    pattern = r'\(([^)]+)\)(\d+\.?\d*)|([A-Z][a-z]*)(\d*\.?\d*)'
    # Parse the entire formula
    for match in re.finditer(pattern, formula):
        if match.group(1):
            inner_str = match.group(1)
            total_count = float(match.group(2))
            # Parse elements inside the parentheses
            inner_pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
            for inner_match in re.finditer(inner_pattern, inner_str):
                elem = inner_match.group(1)
                ratio = float(inner_match.group(2)) if inner_match.group(2) else 1.0
                count = ratio * total_count
                elements[elem] = elements.get(elem, 0) + count
        # Handle single elements 
        elif match.group(3):
            elem = match.group(3)
            count = float(match.group(4)) if match.group(4) else 1.0
            elements[elem] = elements.get(elem, 0) + count
    return elements
def calculate_percentages(elements):
    """Compute the atomic fraction of each element"""
    total_atoms = sum(elements.values())
    return {elem: (count / total_atoms) for elem, count in elements.items()}
# Define target element columns
element_columns = [
    'Li', 'Be', 'B', 'C', 'Mg', 'Al', 'Si', 'P', 'Ca', 'Sc', 'Ti', 'V',
    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Y', 'Zr', 'Nb', 'Mo',
    'Pd', 'Ag', 'In', 'Sn', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Lu', 'Hf', 'Ta', 'W', 'Pt', 'Au'
]
def process_file(input_file, output_file):
    """Process a CSV file and write the featurized results"""
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['alloy', 'Dmax'] + element_columns
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            alloy = row['alloy']
            dmax = row.get('Dmax', '') 
            try:
                # Parse the formula and compute atomic fractions
                elements = parse_formula(alloy)
                percentages = calculate_percentages(elements)
                # Build the output row
                new_row = {'alloy': alloy, 'Dmax': dmax}
                for elem in element_columns:
                    new_row[elem] = round(percentages.get(elem, 0), 5) 
                writer.writerow(new_row)
            except Exception as e:
                print(f"Error while processing alloy {alloy}: {str(e)}")
                error_row = {'alloy': f"ERROR: {alloy}", 'Dmax': dmax}
                for elem in element_columns:
                    error_row[elem] = 0
                writer.writerow(error_row)
if __name__ == "__main__":
    input_file = 'data.csv'  
    output_file = 'data_processed.csv'  
    process_file(input_file, output_file)
print(f"Processing complete! Results saved to {output_file}")
