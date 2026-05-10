import h5py

def print_h5_contents(file_path: str):
    """
    Open the HDF5 file and print the names of all top-level groups/datasets and their values (or shapes/types).
    """
    with h5py.File(file_path, 'r') as f:
        def print_group(name, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                # If the data volume is too large, only print shapes and data types
                print(f"Dataset: {name}")
                print(f"  shape: {data.shape}, dtype: {data.dtype}")
                # To view specific values, uncomment the next line (note: large arrays may make the console freeze)
                # Print the maximum value of the data corresponding to the name attribute
                print(f"  max value: {data.max()}")
                
                print()

        # Iterate over all objects
        f.visititems(print_group)

if __name__ == "__main__":
    h5_path = r"./bgtest_50.h5"
    print_h5_contents(h5_path)
