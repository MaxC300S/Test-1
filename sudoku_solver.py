<# Import all necessary modules
import tkinter as tk
import os
from tkinter import ttk, Button, filedialog


# Function to detect only numbers are present
def correct_number(P):
    # Command ot allow only empty or a single number
    return (P.isdigit() and 1 <= int(P) <= 9) or P == ""


# Create a window
root = tk.Tk()
root.title("9x9 Sudoku Grid")


# Create a Tkinter variable for each cell
cells = [[tk.StringVar() for _ in range(9)] for _ in range(9)]


# Check typed text is a single number
vcmd = (root.register(correct_number), '%P')




# Make the grid
for i in range(9):
    for j in range(9):
        entry = ttk.Entry(root, textvariable=cells[i][j], width=5, justify='center', validate='key', validatecommand=vcmd)
        entry.grid(row=i, column=j, sticky='nsew')
    root.grid_columnconfigure(i, weight=1)


# Function to extract grid values
def extract_grid():
    grid = []
    for row in cells:
        grid_row = []
        for cell in row:
            value = cell.get()
            grid_row.append(int(value) if value.isdigit() else 0)
        grid.append(grid_row)
    return grid




# Function to check if placing num at grid[row][col] is valid
def is_valid(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num or grid[x][col] == num or grid[row - row % 3 + x // 3][col - col % 3 + x % 3] == num:
            return False
    return True


# Solve Sudoku using backtracking
def solve_sudoku(grid):
    empty = find_empty_location(grid)
    if not empty:
        return True
    row, col = empty
    for num in range(1, 10):
        if is_valid(grid, row, col, num):
            grid[row][col] = num
            if solve_sudoku(grid):
                return True
            grid[row][col] = 0
    return False


# Find an empty cell in the grid
def find_empty_location(grid):
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                return (i, j)
    return None



# Update GUI with the solution
def update_gui_with_solution(grid):
    for i in range(9):
        for j in range(9):
            cells[i][j].set(str(grid[i][j]))





# Add functionality to sudoku button
def solve_and_update():
    grid = extract_grid()
    if solve_sudoku(grid):
        update_gui_with_solution(grid)
    else:
        print("No solution exists")



# Import a sudoku grid in a specific format (original purpose was for a OCR grid to be imported)
def import_sudoku():
    try:
        grid = []
        # Construct the file path relative to the script's location (The file will need to be within the same folder as the program)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, "sudoku.txt")
        
        with open(file_path, "r") as file:
            for line in file:
                # Assuming the Sudoku is stored with each number separated by spaces
                row = [int(num) for num in line.strip().split()]
                grid.append(row)
        
        # Assuming 'cells' is a 9x9 grid of tkinter StringVar or similar
        for i in range(9):
            for j in range(9):
                cells[i][j].set("" if grid[i][j] == 0 else str(grid[i][j]))
    except FileNotFoundError:
        print("Error: 'sudoku.txt' file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Import a sudoku grid from a selectable location
def import_external_sudoku():
    try:
        root = tk.Tk()
        root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
        file_path = filedialog.askopenfilename()  # Show an "Open" dialog box and return the path to the selected file
        if not file_path:  # Check if the user canceled the dialog
            return
        
        grid = []
        with open(file_path, "r") as file:
            for line in file:
                row = [int(num) for num in line.strip().split()]
                grid.append(row)
        
        for i in range(9):
            for j in range(9):
                cells[i][j].set("" if grid[i][j] == 0 else str(grid[i][j]))
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")



# Export the sudoku in a grid format to the prohgram path
def export_sudoku():
    grid = extract_grid()
    script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
    file_path = os.path.join(script_dir, "sudoku.txt")  # Construct the file path
    with open(file_path, "w") as file:
        for row in grid:
            line = " ".join(str(num) for num in row) + "\n"  # Added newline character
            file.write(line)
    print(f"Exported sudoku to {file_path}")


# Export the sudoku in grid format to a selectable path
def external_export_sudoku():
    grid = extract_grid()
    root = tk.Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if not file_path:  # Check if the user canceled the dialog
        return

    with open(file_path, "w") as file:
        for row in grid:
            line = " ".join(str(num) for num in row) + "\n"  # Added newline character
            file.write(line)
    print(f"Exported sudoku to {file_path}")



# Clear the grid
def clear_grid():
    for i in range(9):
        for j in range(9):
            cells[i][j].set("")




# Create solve sudoku button
solve_button = ttk.Button(root, text="Solve Sudoku", command=solve_and_update)
solve_button.grid(row=9, column=0, columnspan=9, sticky='nsew')


# Create import button
import_button = ttk.Button(root, text="Import Sudoku", command=import_sudoku)
import_button.grid(row=10, column=0, columnspan=9, sticky='nsew')


# Create export button
export_button = ttk.Button(root, text="Export Sudoku", command=export_sudoku)
export_button.grid(row=12, column=0, columnspan=9, sticky='nsew')


# Create custom import button
import_button = ttk.Button(root, text="Import External Sudoku", command=import_external_sudoku)
import_button.grid(row=11, column=0, columnspan=9, sticky='nsew')


# Create custom export button
export_button = ttk.Button(root, text="Select Sudoku Export path", command=external_export_sudoku)
export_button.grid(row=13, column=0, columnspan=9, sticky='nsew')


# Create clear grid button
clear_button = ttk.Button(root, text="Clear Grid", command=clear_grid)
clear_button.grid(row=14, column=0, columnspan=9, sticky='nsew')


# Create a loop for the window to work
root.mainloop()