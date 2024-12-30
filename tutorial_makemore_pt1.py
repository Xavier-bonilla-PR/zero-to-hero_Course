
###############
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
#######
#Makemore:

N = torch.zeros((27,27), dtype=torch.int32) #makes an array of 28x28 0s
chars = sorted(list(set(''.join(words)))) #list of the 26 letters in a list
stoi = {s:i+1 for i,s in enumerate(chars)} #Creates dictionary mapping each character to index
stoi['.'] = 0 
itos = {i:s for s,i in stoi.items()}
for w in words: 
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1 

plt.figure(figsize=(20,20))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j] 
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray', fontsize=6) #displays letter pair 'ab'
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray', fontsize=6) #displays count of the pair. N[1,j] is a tensor so you need .item() to get the number out
plt.axis('off')


#probability vector
p = N[0].float() #float because we want a prob distribution
p = p / p.sum() # value/total

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
x = itos[ix] 

p = torch.rand(3, generator=g) #creates 3 random numbers between 0,1
p = p / p.sum() #create prob distr

P = (N+1).float()
P /= P.sum(1, keepdims=True)


for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

log_likelihood = 0.0
n = 0
for w in words[:3]: 
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}') 


#######
#Neural network:

#Create the training set of all the bigrams (x,y) 
chars = sorted(list(set(''.join(words)))) 
stoi = {s:i+1 for i,s in enumerate(chars)} 
stoi['.'] = 0 
itos = {i:s for s,i in stoi.items()}

#Create input data and label (what char comes next)
xs, ys = [], []
for w in words: 
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs) 
ys = torch.tensor(ys)

#prints number of elements
num = xs.nelement() 
print('num of examples: ', num)

#Initialize weight and computational graph of operations
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True)  #27 inputs, 27 neurons 



for k in range(50):
    #Foward pass:
    #Convers input to one_hot
    xenc = F.one_hot(xs, num_classes=27).float()
    #Convert one hot to probabilities and normalize it
    logits = xenc @ W 
    counts = logits.exp() 
    probs = counts /counts.sum(1, keepdims=True)
    #loss
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    print(loss.item())
    #reset gradients
    W.grad = None 
    #backwardpropagation
    loss.backward()
    #Update weights
    W.data += -50 * W.grad

#Get names
for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W 
        counts = logits.exp() 
        p = counts / counts.sum(1, keepdims=True) 
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

