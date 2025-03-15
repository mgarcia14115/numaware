import os

# Set the path to your directory containing label files
label_dir = "/home/mgarcia/Desktop/labels"

# Loop through all files in the directory
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_dir, filename)
        
        # Open the file and read the lines
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Create a new list to store updated lines
        updated_lines = []

        # Iterate through each line and swap labels
        for line in lines:
            parts = line.split()
            if parts:
                # Swap the label (0 -> 2, 1 -> 0, 2 -> 1)
                label = int(parts[0])
                if label == 0:
                    parts[0] = '2'
                elif label == 1:
                    parts[0] = '0'
                elif label == 2:
                    parts[0] = '1'
                
                updated_lines.append(" ".join(parts) + "\n")

        # Write the updated lines back to the file
        with open(file_path, "w") as file:
            file.writelines(updated_lines)

print("Label files updated successfully!")

        