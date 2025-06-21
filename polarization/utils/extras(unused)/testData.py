import scipy.io as sio

# Load the .mat file
data = sio.loadmat('sampleData.mat')

# Open a file to write the output
with open('output.txt', 'w') as f:
    # Print all available keys (variables) in the .mat file and write to the file
    f.write("Available keys in the .mat file:\n")
    keys = list(data.keys())
    f.write('\n'.join(keys) + '\n')
    
    # Loop through all keys and print their shapes and full contents to the file
    for key in keys:
        f.write(f"\nKey: {key}\n")
        value = data[key]
        
        # Print the shape of the variable
        f.write(f"Shape: {value.shape if hasattr(value, 'shape') else 'No shape attribute'}\n")
        
        # If the value has a shape (i.e., is an array), print the full contents
        if hasattr(value, 'shape') and value.shape:
            f.write(f"Full contents:\n{value}\n")
        else:
            f.write(f"Value: {value}\n")
            
    f.write("\nEnd of file.")
    
# Print a confirmation message
print("Data has been written to output.txt.")
