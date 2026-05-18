def read_wfx_info(path, wfx_filename=None):
    """
    Searches for a .wfx file in the given path or the previous path, reads the coordinates and charges,
    and stores them in molinfo["geom"].

    :param path: Directory path containing the file.
    :type path: str
    :param wfx_filename: Optional filename of the .wfx file.
    :type wfx_filename: str
    :returns: NumPy array with coordinates.
    :rtype: numpy.ndarray
    """
    global wfxnotfound

    # Normalize paths
    if not path:
        path = os.getcwd()
    path = os.path.normpath(path)
    abs_path = os.path.abspath(path)
    parent_path = os.path.dirname(abs_path)

    # Possible locations to look for the file
    locations = [path, abs_path, parent_path, os.getcwd()]
    # Remove duplicates while preserving order
    unique_locations = []
    for loc in locations:
        if loc and loc not in unique_locations:
            unique_locations.append(loc)

    wfx_file = None

    # If we have a specific filename, look for it
    if wfx_filename:
        # It might be an absolute path already
        if os.path.isabs(wfx_filename) and os.path.isfile(wfx_filename):
            wfx_file = wfx_filename
        else:
            # Look for this specific filename in all locations
            base_wfx = os.path.basename(wfx_filename)
            for loc in unique_locations:
                if loc and os.path.isdir(loc):
                    candidate = os.path.join(loc, base_wfx)
                    if os.path.isfile(candidate):
                        wfx_file = candidate
                        break

    # Fallback to searching for any .wfx file if not found yet
    if not wfx_file:
        for loc in unique_locations:
            if loc and os.path.isdir(loc):
                files = [f for f in os.listdir(loc) if f.lower().endswith(".wfx")]
                if len(files) == 1:
                    wfx_file = os.path.join(loc, files[0])
                    break
                elif len(files) > 1:
                    # If multiple, maybe one matches the "path" name?
                    basename = os.path.basename(path).replace("_atomicfiles", "")
                    matches = [f for f in files if basename in f]
                    if len(matches) == 1:
                        wfx_file = os.path.join(loc, matches[0])
                        break
    
    if not wfx_file:
        if not wfxnotfound:
            print(f" | Could not find .wfx file in any of: {unique_locations}")
            wfxnotfound = True
        return None

    with open(wfx_file, "r") as file:
        lines = file.readlines()

    start_coords = False
    coordinates = []

    for line in lines:
        if "<Atomic Positions>" in line:
            start_coords = True
            continue
        if "</Atomic Positions>" in line:
            start_coords = False
            break

        if start_coords:
            parts = line.split()
            if len(parts) == 4:
                coordinates.append([float(x) for x in parts])

    if not coordinates:
        raise ValueError(f"No coordinates found in the .wfx file: {wfx_file}")

    # Combine symbols and coordinates into a NumPy array
    geom = np.array([coordinates[i] for i in range(len(coordinates))], dtype=object)

    return geom
