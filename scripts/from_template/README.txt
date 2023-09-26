# XYZ to Python Converter

This tool automates the conversion of **.xyz** files into corresponding Python files using the provided **template.py** as a reference.

## Prerequisites

1. **template.py**: Customize this file according to your specific requirements. You can start with the provided template, but ensure that you adjust the number of lines above the molecular coordinates in **template.py** to match your data format.

## Usage

1. Copy the following files to your working directory:
   - `pythonize.sh`
   - `template.py`

2. Ensure that your **.xyz** files contain only the molecular geometries. Place these **.xyz** files in the same directory where the `pythonize.sh` script is located.

3. Modify the permissions of the `pythonize.sh` script using the following command:
   ```bash
   chmod u+rwx pythonize.sh
   ```
4. Run the pythonize.sh script to perform the conversion:
  ```bash
  ./pythonize.sh
  ```
This script will automatically insert the molecular coordinates into the Python structure according to the template.

## Testing

To test the utility, execute the pythonize.sh script. It will create a Python file (e.g., example.py) with the Python structure derived from the template.py file and the coordinates from the corresponding .xyz file. You can repeat this process for any number of .xyz files you need to convert.
