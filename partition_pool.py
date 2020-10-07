
with open('avgpool.txt', 'r') as f:
    list_archs = eval(f.read())

length = len(list_archs)

pool_1 = []
pool_2 = []

for i in range(len(list_archs)):
    if i < length / 2:
        pool_1.append(list_archs[i])
    else:
        pool_2.append(list_archs[i])


with open('pool_1.txt', 'w') as f:
    f.write(str(pool_1))

with open('pool_2.txt', 'w') as f:
    f.write(str(pool_2))