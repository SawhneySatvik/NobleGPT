import torch
import torch.nn as nn
from torch.nn import functional as F

#hyper parameters
batch_size = 32
chunk_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# read it to inspect it
with open(r'BIGGEST PROJECT\Final\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

#Encoder and decoder
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[stoi[c] for c in s]
decode=lambda l:''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

#Split up data into train and validation set
n=int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data laoding
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - chunk_size, (batch_size,))
    x = torch.stack([data[i:i+chunk_size] for i in ix])
    y = torch.stack([data[i+1:i+chunk_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()#to tell pytorch that everything in this function doesnot call backward(make it more memory efficient)
def estimate_loss():#finds average mean loss over multiple batchs so that the loss is much less noisy
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)#key vector show what does it contain before 
        self.query = nn.Linear(n_embd, head_size, bias=False)#query vector show what we want to predict
        self.value = nn.Linear(n_embd,head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(chunk_size,chunk_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,16) cause head size is 16
        q = self.query(x) #(B,T,16)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 #(B,T,16) @ (B,16,T) ----> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))#a triangular matrix is made becasue for ex the 2nd token can take into account the future tokens meaning weights after that i.e. 3rd 4th 5th 6th 7th and 8th
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)# projection is just linear transformation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):#per toekn level
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)#normalize layer
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

#very simplified bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(chunk_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)#final layer norm done after sa and ffwd
        self.lm_head = nn.Linear(n_embd,vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) #(B,T,C) B= batch size T= time or chunk size C = channel or vocab size
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T, C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) #(B,T, vocab size)

        if targets is None:
            loss=None
        else:
            B, T, C=logits.shape
            logits=logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits,loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last chunk tokens
            idx_cond = idx[:,-chunk_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:,-1, :] # becomes (B, C) checks the logits to the last in time dimnsion
            #apply softmax to get probabilities
            probs=F.softmax(logits, dim=1) # converts logits into probability
            #sample from distribution
            idx_next=torch.multinomial(probs,num_samples=1) #(B, 1)
            #appending the sampled index to the running sequence
            idx = torch.cat((idx,idx_next), dim=1) #(B, T+1) in essense joins the predicted to the original then continues predicting
        return idx
    
model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
'''
#training the model
#creating a PyTorch optimizer
optmizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    #every once in a while evaluates the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f},val loss {losses['val']:.4f}")

    #sample a batch of data
    xb, yb=get_batch('train')

    #evaluate loss
    logits, loss = m(xb,yb)
    optmizer.zero_grad(set_to_none=True)
    loss.backward()
    optmizer.step()

torch.save(model.state_dict(), 'model.pt')
'''

#generate from model
m=model.load_state_dict(torch.load('model.pt'))
m = model.to(device)

#Creating prompt and encoding it
prompt = input('Type a prompt: ')
context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

#generating from the model
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
#open(r'BIGGEST PROJECT\Final\more1.txt', 'w').write(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))
