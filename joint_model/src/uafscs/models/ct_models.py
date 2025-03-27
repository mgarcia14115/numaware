import torch


class regression_with_midpoints_pose(torch.nn.Module):
    def __init__(self):
        super(regression_with_midpoints_pose, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3)
        )

    def forward(self, image, mid, **kwargs):
        return self.seq(mid)


class regression_with_images_midpoints(torch.nn.Module):
    def __init__(self):
        super(regression_with_images_midpoints, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, (3,3), 1, 1),
            torch.nn.MaxPool2d((2,2), 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, (3,3), 1, 1),
            torch.nn.MaxPool2d((2,2), 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, (3,3), 1, 1),
            torch.nn.MaxPool2d((2,2), 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, (3,3), 1, 1),
            # torch.nn.MaxPool2d((2,2), 2, 1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(256, 256, (3,3), 1, 1),
            # torch.nn.MaxPool2d((2,2), 2, 1),
            # torch.nn.ReLU(),
            
        )

        self.lin = torch.nn.Sequential(
            torch.nn.Linear(86018, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32, 3),
        )

        

    def forward(self, image, mid, **kwargs):
        x = self.conv(image)
        x = torch.flatten(x, start_dim=1) 
        x = torch.cat((x, mid), 1)
        x = self.lin(x)
        return x



# ============================================================================================================================

class PatchEmbedding(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# 2. Adding Positional Embeddings
class PositionalEncoding(torch.nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.pos_embed = torch.nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embed
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        return self.attn(x, x, x)[0]
    
class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=3, embed_dim=768, num_heads=8, depth=6, mlp_dim=1024):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, (img_size // patch_size) ** 2)
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mlp_head = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x, mid):
        B = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.mlp_head(x[:, 0])
