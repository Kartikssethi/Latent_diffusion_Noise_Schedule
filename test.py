#DefaultDict and Normal Dict
from collections import defaultdict

a = defaultdict(int)

for i in range(5):
    a[i] += 1
    
b = {}
for i in range(5):
    if b[i]:
        b[i]+= 1
    else:
        b[i] = 1