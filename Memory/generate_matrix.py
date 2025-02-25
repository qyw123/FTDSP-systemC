import random

def generate_matrix(file_path, rows, cols, value_range=(1.0, 10.0)):
    """
    Generates a matrix with random floating-point numbers and writes it to a file.

    Args:
        file_path (str): Path to the output file.
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        value_range (tuple): Range of random values (min, max).
    """
    with open(file_path, "w") as file:
        for _ in range(rows):
            row = [f"{random.uniform(*value_range):.2f}" for _ in range(cols)]
            file.write(" ".join(row) + "\n")

def generate_zero_matrix(file_path, rows, cols):
    with open(file_path, "w") as file:
        for _ in range(rows):
            row = [0.0 for _ in range(cols)]
            file.write(" ".join(map(str, row)) + "\n")

if __name__ == "__main__":
    output_file = "matrixC_input.txt"
    num_rows = 12  # Set the number of rows
    num_cols = 128  # Set the number of columns
    #generate_matrix(output_file, num_rows, num_cols)
    generate_zero_matrix(output_file, num_rows, num_cols)

    print(f"Matrix saved to {output_file}")
