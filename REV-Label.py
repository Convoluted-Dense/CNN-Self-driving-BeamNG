import matplotlib.pyplot as plt

# Read data from text file
file_path = "dataset/modified_steering.txt"  # Update with your file path
with open(file_path, "r") as file:
    lines = file.readlines()

# Parse data
values = []
for line in lines:
    img_name, value = line.split()
    values.append(float(value))

# Plotting histogram
plt.figure(figsize=(20, 6))
plt.hist(values, bins=100, color='skyblue', edgecolor='black')
plt.title('Histogram of Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()