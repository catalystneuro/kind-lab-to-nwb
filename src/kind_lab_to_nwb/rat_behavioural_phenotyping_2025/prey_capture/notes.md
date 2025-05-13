# Notes concerning the prey capture conversion

## USV detections .mat file conversion script

The detection files need to be converted to ensure compatibility with Python. This can be done in two ways:

### 1. Using MATLAB directly
The MATLAB script `convert_detections_mat_files_to_v73.m` converts the "Calls" table to struct to ensure readability in Python.

### 2. Using Python to execute MATLAB commands
Alternatively, you can use the Python script `run_matlab_command.py` to execute the conversion directly from Python. This approach:
- Calls MATLAB from Python using subprocess
- Requires MATLAB to be installed on the system
- Executes the same conversion logic but through Python
- Creates a 'converted' subdirectory automatically

Both methods perform the following operations:
- Convert the "Calls" table to struct format for Python compatibility
- Save files in MATLAB v7.3 format
- Create a 'converted' subdirectory to store the processed files
- Maintain original data integrity

### Usage
1. **Pre-conversion Step**
   - Must be run before the Python-based NWB conversion pipeline
   - Processes all .mat files in specified directories and subdirectories

2. **Running the Conversion**
   - Using MATLAB: Run the `convert_detections_mat_files_to_v73.m` script directly
   - Using Python: Run `run_matlab_command.py` with appropriate MATLAB path and folder path parameters

3. **Output**
   - Converted files are saved in a 'converted' subdirectory
   - Original files remain unchanged
   - Python-compatible .mat files in v7.3 format
