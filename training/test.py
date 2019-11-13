import torch

size = 5
x = (torch.triu(torch.ones(size, size))).transpose(0, 1)
#print(x)
for i in range(size):
    x = (torch.triu(torch.ones(size, size))).transpose(0, 1)

    for m in range(i+1,size):
        #print(m)
        x[m] = 0

    #print(x)


x = (torch.triu(torch.ones(size, size))).transpose(0, 1)
print(x)
for i in range(1,5):
    print(x[:i,:i])
    print((torch.triu(torch.ones(i, i))).transpose(0, 1))