import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.

    Args:
        img_size (int): Size of the image (it is a square)
        patch_size (int): Size of the patch (it is a square)
        in_chans (int): Number of input channels
        embed_dim (int): The embedding dimension

    Attributes:
        n_patches (int): Number of patches inside of our image
        proj (nn.Conv2d): Convolutional layer that does both the splitting into patches and their embedding
    """

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels = in_chans,
            out_channels = embed_dim, # 각각의 patch가 가질 embedding dimension
            kernel_size=patch_size, # kernel size를 patch_size(16*16)으로 만들어서 각 patch당 embedding을 만든다
            stride=patch_size, # stride를 patch_size로 지정해서 kernel끼리 서로 overlapping되지 않도록 설정
        )

    def forward(self, x):
        """Run forward pass.

        Args:
            x (torch.Tensor): Shape `(n_samples, in_chans, img_size, img_size)` -> (batch, channel, height, width)

        Returns:
            (torch.Tensor): Shape `(n_samples, n_patches, embed_dim)
        """
        x = self.proj(x) # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(start_dim=2) # (n_samples, embed_dim, n_patches)
        x = x.transpose(1,2) # (n_samples, n_patches, embed_dim)

        return x


class Attention(nn.Module):
    """Attention mechanism.
    
    Args:
        dim (int): The input and out dimension of per token features
        n_heads (int): Number of attention heads
        qkv_bias (bool): If True then we include bias to the query, key and value projections
        attn_p (float): Dropout probability applied to the query, key and value tensors.
        proj_p (float): Dropout probability applied to the output tensor

    Attributes:
        scale (float): Normalizing constant for the dot product
        qkv (nn.Linear): Linear projection for the query, key and value
        proj (nn.Linear): Linear mapping that takes in the concatenated output of all attention heads and maps it into a new space
        attn_drop, proj_drop (nn.Dropout): Dropout layers
    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        self.head_dim = dim // n_heads # embedding dimension은 head의 수에 비례
        self.scale = self.head_dim ** (-0.5) # 각 head에서 scaled dot product를 위한 scaling (dk = head_dim)
        self.qkv = nn.Linear(in_features=dim, out_features=dim*3, bias=qkv_bias) # 3배를 하는 이유는 self-attention이기 때문에 qkv를 한번에 연산
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    
    def forward(self, x):
        """Run forward pass

        Args:
            x (torch.Tensor): Shape `(n_samples, n_patches + 1, dim)` -> classification token까지 포함한 shape
        
        Returns:
            torch.Tensor: Shape `(n_samples, n_patches + 1, dim)`
        """

        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim) # (n_samples, n_patches + 1, 3, heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2] # each query, key, and value : (n_samples, n_heads, n_patches + 1, head_dim)
        k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1) # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_Drop(attn)

        weighted_avg = attn @ v # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1,2) # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg) # (n_samples, n_patches + 1, dim) -> head별로 연산된 attention을 새로운 vector space로 projection 시키기 위해서 마지막에 필요!
        x = self.proj_drop(x) # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    """Multilayer perceptron

    Args:
        in_features (int): Number of input features
        hidden_features (int): Number of nodes in the hidden layers
        out_features (int): Number of output features
        p (float): Dropout probability
    
    Attributes:
        fc (nn.Linear): The First linear layer
        act (nn.GELU): GELU activation function (Gaussain Error Linear Unit)
        fc2 (nn.Linear): The second linear layer
        drop (nn.Dropout): Dropout layer
    """

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass

        Args:
            x (torch.Tensor): Shape `(n_samples, n_patches + 1, in_features)`
        
        Returns:
            torch.Tensor: Shape `(n_samples, n_patches + 1, out_features)`  # in feature랑 out feature를 같게 해서 skip connection이 가능하도록!
        """
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    """Transformer block

    Args:
        dim (int): Embedding dimension
        n_heads (int): Number of attention heads
        mlp_ratio (float): Determines the hidden dimension size of the 'MLP' module with respect to 'dim'
        qkv_bias (bool): If True then we include biase to the query, key, and value projections
        proj_p, attn_p (float): Dropout probability
    
    Attributes:
        norm1, nomr2 (LayerNorm): Layer normalization
        attn (Attention): Attention module
        mlp (MLP): MLP module
    """

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, proj_p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=proj_p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        """Run forward pass

        Args:
            x (torch.Tensor): Shape `(n_samples, n_patches + 1, dim)`

        Returns:
            torch.Tensor: Shape `(n_samples, n_patches + 1, dim)`
        """

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """Simplified implementation of the Vision Transformer

    Args:
        img_size (int): Both height and the width of the image (it is a square)
        patch_size (int): Both height and the width of the patch (it is a square)
        in_chans (int): Number of input channels
        n_classes (int): Number of classes
        embed_dim (int): Dimensionality of the token/patch embeddings
        depth (int): Number of blocks
        n_heads (int): Number of attention heads
        mlp_ratio (float): Determine the hidden dimension of Feed forward network
        qkv_bias (bool): If True then we include bias to the query, key, and value projections
        p, attn_p (float): Dropout probability

    Attributes:
        patch_embed (PatchEmbed): Instance of 'PatchEmbed' layer
        cls_token (nn.Parameter): Learnable paramter that will represent the first token in the sequence. It has 'embed_dim' elements
           - nn.Parameter: A kind of Tensor that is to be considered a module parameter.
        pos_emb (nn.Parameter): Positional embedding of the cls token + all the patches. It has '(n_patches + 1) * embed_dim' elements.
        pos_drop (nn.Dropout): Dropout layer
        blocks (nn.ModuleList): List of 'Block' modules
        norm (nn.LayerNorm): Layer normalization
    """

    def __init__(
        self,
        img_size=384,
        patch_size=16,
        in_chans=3,
        n_classes=1000,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        p=0.,
        attn_p=0.,
    ):

        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.clss_head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """Run the forward pass

        Args:
            x (torch.Tensor): Shape `(n_samples, in_chans, img_size, img_size)`
        
        Returns:
            logits (torch.Tensor): Logits over all the classes - `(n_samples, n_classes)`
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1) # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim = 1) # (n_samples, n_patches + 1, embed_dim)
        x = x + self.pos_embed # (n_samples, 1 + n_patches, embed_dim) -> automatically broadcasting

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        cls_token_final = x[:, 0] # [CLS] embedding
        x = self.clss_head(cls_token_final)
        return x

