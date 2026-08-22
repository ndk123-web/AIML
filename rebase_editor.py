import sys

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

lines[0] = lines[0].replace('pick', 'edit')

with open(sys.argv[1], 'w') as f:
    f.writelines(lines)
