import os

def correct_labels(directory):
    # Iterate through each label file in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Make sure it's a label file
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Replace the incorrect class labels
            corrected_lines = []
            for line in lines:
                # Split the line into components
                components = line.split()
                label = int(components[0])  # The class label is the first component
                
                # Change the class labels according to the mapping
                if label == 0:
                    components[0] = '1'  # 0 (white) -> 1 (grey)
                elif label == 1:
                    components[0] = '0'  # 1 (grey) -> 0 (white)
                # If it's 2 (black), leave it unchanged
                
                corrected_lines.append(' '.join(components) + '\n')
            
            # Write the corrected lines back to the file
            with open(file_path, 'w') as file:
                file.writelines(corrected_lines)

            print(f"Updated labels in: {filename}")

# Replace with the directory containing your label files
directory = "/home/mgarcia/Desktop/labels"
correct_labels(directory)
