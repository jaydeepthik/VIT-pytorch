import torch
import torch.nn as nn

import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm, trange


np.random.seed(42)
torch.manual_seed(42)

def get_patches(x, num_patches):
    N, C, H, W = x.shape
    assert H == W

    patches = torch.zeros(N, num_patches**2, C*H*W//num_patches**2)
    patch_size = H//num_patches

    for idx, img in enumerate(x):
        for i in range(num_patches):
            for j in range(num_patches):
                patch = img[:,i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                patches[idx,i*num_patches +j] = patch.flatten()
    return patches


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads = 2) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        head_dim = int(self.hidden_dim//self.num_heads)
        self.head_dim = head_dim

        self.q = nn.ModuleList([nn.Linear(hidden_dim, head_dim) for _ in range(self.num_heads)])
        self.k = nn.ModuleList([nn.Linear(hidden_dim, head_dim) for _ in range(self.num_heads)])
        self.v = nn.ModuleList([nn.Linear(hidden_dim, head_dim) for _ in range(self.num_heads)])

        self.sftmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x (N, P, D)
        result = []
        for inp in x: #(P, D)
            inp_result = []
            for head in range(self.num_heads):
                q_map = self.q[head]
                k_map = self.k[head]
                v_map = self.v[head]

                q, k, v = q_map(inp), k_map(inp), v_map(inp) # (P, HD)
                attn = self.sftmax(q@k.T/(self.head_dim)) # (P, HD) @ (HD, P) -----> (P, P)
                inp_result.append(attn @ v) # (P, HD)
            result.append(torch.hstack(inp_result)) #(P, D)

        return torch.cat([torch.unsqueeze(r, dim=0) for r in result]) #(N, P, D)
    

class ViTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_raio=4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_raio

        self.l_norm1 = nn.LayerNorm(self.hidden_dim)
        self.mhsa = MultiHeadedAttention(hidden_dim=self.hidden_dim, num_heads=2)
        self.l_norm2 = nn.LayerNorm(self.hidden_dim)
        self.MLP = nn.Sequential(
            nn.Linear(self.hidden_dim, mlp_raio*self.hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_raio*self.hidden_dim, self.hidden_dim)
        )

    def forward(self, x):
        out = self.mhsa(self.l_norm1(x)) + x
        out = out + self.MLP(self.l_norm2(out))
        return out

class ViT(nn.Module):
    def __init__(self, img_shape = (1, 28, 28), num_patches = 7, hidden_d = 8, num_heads = 2, num_blocks=8, out_d=10) -> None:
        super().__init__()

        self.img_shape = img_shape
        self.num_patches = num_patches
        self.hidden_d = hidden_d
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.patch_size = self.img_shape[2]//self.num_patches
        self.input_d = self.patch_size**2

        #input to linear tokens
        self.linear_map = nn.Linear(16, self.hidden_d)

        #cls_token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        #positional embeddings
        self.register_buffer('pos_emb', get_positional_embeddings(self.num_patches ** 2 + 1, hidden_d), persistent=False)
        self.pos_emb.requires_grad = False

        #transformer blocks
        self.vitblocks = nn.ModuleList([ViTBlock(self.hidden_d, self.num_heads) for _ in range(self.num_blocks)])

        #clf
        self.clf = nn.Sequential(
            nn.Linear(self.hidden_d ,out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        patches = get_patches(x, self.num_patches).to(self.pos_emb.device)
        tokens = self.linear_map(patches)

        #add cls_token
        tokens = torch.cat((self.class_token.expand(x.shape[0], 1, -1), tokens), dim=1)

        #add pos_emb
        pos_emb = self.pos_emb.repeat(x.shape[0], 1, 1)
        out = tokens + pos_emb

        #Transformer encoder
        for block in self.vitblocks:
            out = block(out)

        #clf token
        out = out[:,0]
        out = self.clf(out)    

        return out
    
if __name__ == "__main__" :
    #prepare data
    transform = ToTensor()

    train_data = MNIST("./datasets",train=True, download=True, transform=transform)
    test_data = MNIST("./datasets",train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=128)
    test_dataloader = DataLoader(train_data, shuffle=False, batch_size=128)

    #training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    #model
    vit = ViT(img_shape=(1, 28, 28), num_patches=7, hidden_d=8, num_heads=2, num_blocks=8, out_d=10).to(device)
    num_epochs = 10
    lr= 1e-3

    #train
    optimizer = Adam(vit.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    for epoch in trange(num_epochs, desc="Training"):
        train_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)

            preds = vit(x)
            loss = criterion(preds, y)
            train_loss += loss.detach().cpu().item() / len(train_dataloader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_dataloader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = vit(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_dataloader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")