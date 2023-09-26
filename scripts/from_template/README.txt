
By running the bash script **pythonize.sh** (as ./pythonize.sh), all **.xyz** files will be converted
into the corresponding python files using the **template.py** file as
reference. The proper way to use these tools is to copy the **pythonize.sh**
and **template.sh** files to the working directory.

- The **template.py** file needs to be customized according to the author's
  necessities (feel free to use the provided templates, but remember to adapt the number of
  lines above the molecular coordinates in **template.py**).

- The **.xyz** files should contain only the geometries. They need to be located in
  the same directory where the **pythonize.sh** script is.

- The **pythonize.sh** script can be altered in the working directory by using
  ** chmod u+rwx pythonize.sh **. When ran, it will insert all the coordinates
  as the geometries according to the template.

To test the utility, run the pythonize.sh script and it will create an
example.py with the Python structure from the template.py file and the
coordinates from the example.xyz. There is no limit on how many .xyz need to
be created.

