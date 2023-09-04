#!/bin/bash

for pyfile in *.xyz; do
    # Create the Python file corresponding to the title
    pyfile="${pyfile%.xyz}.py"
    touch "${pyfile}"

    # Copy the first 8 lines of test.py to the new Python file
    sed -n '1,7p' template.py > "${pyfile}"
    
    # Change the value of molname to the title
    title="${pyfile%.py}"
    sed -i "s/molname = ''/molname = '${title}'/" "${pyfile}"

    # Append the contents of the title.xyz file to the Python file
    cat "${pyfile%.py}.xyz" >> "${pyfile}"

    # Append the contents of test.py from line 9 to the end to the Python file
    sed -n '8,$p' template.py >> "${pyfile}"

 done
