import torch
import torch.nn as nn
import math


class MzEmbeddings(nn.Module):
    def __init__(self, embed_dim: int, total_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.total_size = total_size
        self.embedding = nn.Embedding(total_size, embed_dim)

    def forward(self, mz_idx, intensity):
        # (batch, seq_len) --> (batch, seq_len, embed_dim)
        embedding = self.embedding(mz_idx)
        norm = embedding.norm(p=2, dim=1, keepdim=True)
        embedding = embedding / norm
        embedding = embedding * intensity.unsqueeze(2)
        return embedding
    
class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 6000, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        

class FeedForwardBlock(nn.Module):

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, ff_dim) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, embed_dim) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, embed_dim) --> (batch, seq_len, d_ff) --> (batch, seq_len, embed_dim)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, h: int, dropout: float) -> None:
        """
        Multi-Head Attention Block.

        Args:
            embed_dim (int): The dimension of the input embedding.
            h (int): The number of attention heads.
            dropout (float): Dropout rate for attention scores.
        """
        super().__init__()
        self.embed_dim = embed_dim  # Embedding vector size
        self.h = h  # Number of heads
        
        # Ensure the embedding dimension is divisible by the number of heads
        assert embed_dim % h == 0, "embed_dim must be divisible by h"
        
        self.d_k = embed_dim // h  # Dimension of each head's vector

        # Linear layers to project the input into Q, K, and V
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)  # Linear layer for query
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)  # Linear layer for key
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)  # Linear layer for value

        # Output linear layer
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=False)  # Linear layer for the final output

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        Calculate the scaled dot-product attention.
        
        Args:
            query (torch.Tensor): The query matrix Q.
            key (torch.Tensor): The key matrix K.
            value (torch.Tensor): The value matrix V.
            mask (torch.Tensor, optional): The mask to prevent attention to certain positions.
            dropout (nn.Dropout, optional): Dropout layer for attention scores.

        Returns:
            torch.Tensor: The attention-weighted output.
            torch.Tensor: The attention scores.
        """
        d_k = query.shape[-1]  # Dimension of each head
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (if provided)
        if mask is not None:
            # Expand mask to match the attention scores dimensions
            expand_mask = mask.unsqueeze(1).unsqueeze(2).transpose(2, 3).expand(attention_scores.shape)
            attention_scores = attention_scores.masked_fill(expand_mask == True, -1e9)

        # Apply softmax to normalize the scores
        attention_scores = torch.softmax(attention_scores, dim=-1)

        # Apply dropout (if provided)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Compute the attention-weighted output
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, q, k, v, mask=None):
        """
        Forward pass of the Multi-Head Attention block.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (torch.Tensor, optional): Mask tensor to apply on attention scores.

        Returns:
            torch.Tensor: Output tensor after multi-head attention.
        """
        # Linear projections for Q, K, V
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split the embeddings into h heads and reshape (batch_size, seq_len, embed_dim) --> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Compute attention
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # Concatenate all heads back together (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Final linear transformation (batch_size, seq_len, embed_dim)
        return self.w_o(x)


# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.1):
#         super(SelfAttention, self).__init__()
#         self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

#     def forward(self, x):
#         # x: (batch_size, seq_len, embed_dim)
#         attn_output, attn_weights = self.self_attention(x, x, x)
#         # attn_output: (batch_size, seq_len, embed_dim)
#         return attn_output, attn_weights

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttentionBlock(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout)  # FeedForwardBlockを使用
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, embed_dim)
        
        # Self-Attention Block
        attn_output = self.attention(x, x, x, mask)
        # attn_output: (batch_size, seq_len, embed_dim)
        x = self.norm1(x + self.dropout(attn_output))
        # x: (batch_size, seq_len, embed_dim)
        
        # Feed Forward Block
        ff_output = self.feed_forward(x)
        # ff_output: (batch_size, seq_len, embed_dim)
        x = self.norm2(x + self.dropout(ff_output))
        # x: (batch_size, seq_len, embed_dim)
        
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, embed_dim)
        for layer in self.layers:
            x = layer(x, mask)
        # 最後に正規化を適用
        x = self.norm(x)
        # x: (batch_size, seq_len, embed_dim)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        # Self-Attention Block (causal mask is usually applied here in practice)
        self.self_attention = MultiHeadAttentionBlock(embed_dim, num_heads, dropout)
        
        # Encoder-Decoder Attention Block
        self.enc_dec_attention = MultiHeadAttentionBlock(embed_dim, num_heads, dropout)
        
        # Feed Forward Block
        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout)  # FeedForwardBlockを使用
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            x (torch.Tensor): Target sequence input (batch_size, tgt_seq_len, embed_dim)
            enc_output (torch.Tensor): Encoder output (batch_size, src_seq_len, embed_dim)
            tgt_mask (torch.Tensor, optional): Mask for the target sequence (Self-Attention mask).
            memory_mask (torch.Tensor, optional): Mask for the encoder output (Encoder-Decoder Attention mask).
        """
        # Self-Attention Block
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Encoder-Decoder Attention Block
        enc_dec_attn_output = self.enc_dec_attention(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x + self.dropout(enc_dec_attn_output))
        
        # Feed Forward Block
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            x (torch.Tensor): Target sequence input (batch_size, tgt_seq_len, embed_dim)
            enc_output (torch.Tensor): Encoder output (batch_size, src_seq_len, embed_dim)
            tgt_mask (torch.Tensor, optional): Mask for the target sequence (Self-Attention mask).
            memory_mask (torch.Tensor, optional): Mask for the encoder output (Encoder-Decoder Attention mask).
        """
        # x: (batch_size, tgt_seq_len, embed_dim)
        # enc_output: (batch_size, src_seq_len, embed_dim)
        
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        
        # 最後に正規化を適用
        x = self.norm(x)
        # x: (batch_size, tgt_seq_len, embed_dim)
        return x




class Conv1dFlatten(nn.Module):
    def __init__(self, embed_dim, out_features, seq_len, conv_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(Conv1dFlatten, self).__init__()
        
        layers = []
        in_channels = embed_dim
        for out_channels in conv_channels:
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)), 
            layers.append(nn.ReLU())
            in_channels = out_channels
        
        # 最終的な畳み込み層で必要な出力特徴量に合わせる
        layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_features, kernel_size=seq_len))

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        x = self.conv_layers(x) # (batch, out_features, 1)
        x = x.squeeze(2) # (batch_size, out_features, 1) -> (batch_size, out_features)
        return x
    


class LinearFlatten(nn.Module):
    def __init__(self, embed_dim, out_features, hidden_dim:list[int]=[]):
        super(LinearFlatten, self).__init__()
        
        layers = []
        in_channels = embed_dim
        nodes = [embed_dim] + hidden_dim + [out_features]
        for i in range(len(nodes) - 2):
            layers.append(nn.Linear(in_features=nodes[i], out_features=nodes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=nodes[-2], out_features=nodes[-1]))

        self.linear_layers = nn.Sequential(*layers)

    def forward(self, x):
        # Apply average pooling over the sequence length dimension
        x = x.mean(dim=1)  # Shape: [batch_size, embed_dim]

        # Linear transformation: [batch_size, embed_dim] -> [batch_size, out_features]
        x = self.linear_layers(x)
        return x
    
if __name__ == "__main__":
    # Parameters
    batch_size = 2
    seq_len = 5
    embed_dim = 4
    num_heads = 2

    # Sample input: [batch_size, seq_len, embed_dim]
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Key padding mask: [batch_size, seq_len]
    # True = padding position (ignore), False = valid position (attend to)
    key_padding_mask = torch.tensor([[False, False, True, True, True],  # Only first 2 tokens are valid
                                    [False, False, False, True, True]]) # Only first 3 tokens are valid

    # Define multi-head attention
    multi_head_attn = MultiHeadAttentionBlock(embed_dim=embed_dim, h=num_heads, dropout=0.1)

    # Apply multi-head attention with key_padding_mask
    attn_output, attn_weights = multi_head_attn(x, x, x, mask=key_padding_mask)

    # Print results
    print("Attention Output:")
    print(attn_output)

    print("\nAttention Weights:")
    print(attn_weights)