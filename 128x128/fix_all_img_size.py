"""
Comprehensive fix - removes ALL img_size references
"""

# Read inference.py
with open('inference.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Process line by line to remove img_size
new_lines = []
for line in lines:
    # Skip lines that define img_size parameter
    if 'img_size: int' in line and 'def __init__' in ''.join(new_lines[-5:]):
        # Remove img_size from parameter list
        line = line.replace('img_size: int = 128,', '')
        line = line.replace('img_size: int = 64,', '')
        line = line.replace(', img_size: int = 128', '')
        line = line.replace(', img_size: int = 64', '')
    
    # Remove self.img_size assignments
    if 'self.img_size' in line:
        continue  # Skip this line entirely
    
    # Remove img_size from Generator() calls
    if 'Generator(' in line and 'img_size=' in line:
        line = line.replace(', img_size=img_size', '')
        line = line.replace('img_size=img_size,', '')
        line = line.replace('img_size=img_size', '')
        line = line.replace(', img_size=self.img_size', '')
        line = line.replace('img_size=self.img_size,', '')
        line = line.replace('img_size=self.img_size', '')
    
    new_lines.append(line)

# Write back
with open('inference.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ Removed ALL img_size references from inference.py")
print("Now run: py app_enhanced.py")
