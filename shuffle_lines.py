import random, sys
lines = open(sys.argv[1]).readlines()
random.shuffle(lines)
print(''.join(lines))

# python shuffle_lines.py ch3_ex_19.txt