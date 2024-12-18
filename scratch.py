from collections import Counter


x1 = ['COMP2', 'SDKC9', 'COMP3', 'SDKC7CL', 'SDKC7CU', 'TV3', 'TV6', 'STO5135D', 'STO7143D', 'ZC79X']
x2 = ['COMP2', 'COMP3', 'STO5135D', 'STO7143D', 'TV6', 'ZC79X', 'SDKC9', 'SDBB9CU', 'TV4', 'SDKC7CL', 'ATR2']

x3 =['COMP2', 'COMP3', 'STO5135D', 'TV6', 'ZC79X']
x4 = ['COMP2', 'SDKC9', 'COMP3', 'TV6']

# Combine the lists
combined_list = x1 + x2 + x3 + x4

# Count the frequency of each item
item_counts = Counter(combined_list)

# Sort by count (descending) and then by item (alphabetically)
sorted_item_counts = sorted(item_counts.items(), key=lambda x: (-x[1], x[0]))

# Print the sorted counts
print(f" Features used  {len(sorted_item_counts)} ")
print("Item counts:")
for item, count in sorted_item_counts:
    print(f"{item}: {count}")
    
print("[")
for item, count in sorted_item_counts:
    print(f"'{item}',")
print("]")    
