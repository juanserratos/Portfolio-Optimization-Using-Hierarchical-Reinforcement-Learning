"""
Transformer-based model for market state encoding.
"""
import torch
import torch.nn as nn
import math
import logging

# Configure logging
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    """
    def __init__(self, d_model, max_len=1000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Embedding dimension
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (persistent state that's not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]

class MultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention module.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initialize multi-head self-attention.
        
        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super(MultiheadSelfAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            tuple: (Output tensor, attention weights)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores (scaled dot-product attention)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project back to d_model dimension
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        
        return output, attn_weights

class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer.
    """
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        """
        Initialize transformer encoder layer.
        
        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout rate
        """
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiheadSelfAttention(d_model, num_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Self-attention block
        attn_output, attn_weights = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feedforward block
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x, attn_weights

class MarketTransformer(nn.Module):
    """
    Transformer model for market state encoding.
    """
    def __init__(self, feature_dim, hidden_dim, num_heads, num_layers, num_assets, dropout=0.1, max_seq_len=60):
        """
        Initialize market transformer.
        
        Args:
            feature_dim (int): Dimension of input features per asset
            hidden_dim (int): Hidden dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            num_assets (int): Number of assets
            dropout (float): Dropout rate
            max_seq_len (int): Maximum sequence length
        """
        super(MarketTransformer, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_assets = num_assets
        
        # Input embedding layers for each type of data
        self.price_embedding = nn.Linear(5, hidden_dim // 4)  # OHLCV
        self.technical_embedding = nn.Linear(10, hidden_dim // 4)  # Technical indicators
        self.fundamental_embedding = nn.Linear(3, hidden_dim // 4)  # Fundamental metrics
        self.market_embedding = nn.Linear(2, hidden_dim // 4)  # Market-wide features
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim * num_assets, hidden_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized MarketTransformer with {num_layers} layers, "
                   f"{num_heads} heads, and hidden dimension {hidden_dim}")
        
    def _init_weights(self, module):
        """
        Initialize weights of the model.
        
        Args:
            module (nn.Module): Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_assets, feature_dim]
            
        Returns:
            torch.Tensor: Encoded market state
        """
        batch_size, seq_len, num_assets, _ = x.size()
        
        # Split features into different types
        price_data = x[:, :, :, :5]  # OHLCV
        technical_data = x[:, :, :, 5:15]  # Technical indicators
        fundamental_data = x[:, :, :, 15:18]  # Fundamental metrics
        market_data = x[:, :, :, 18:20]  # Market-wide features
        
        # Process each asset and concatenate embeddings
        embeddings = []
        for i in range(num_assets):
            price_emb = self.price_embedding(price_data[:, :, i, :])
            tech_emb = self.technical_embedding(technical_data[:, :, i, :])
            fund_emb = self.fundamental_embedding(fundamental_data[:, :, i, :])
            market_emb = self.market_embedding(market_data[:, :, i, :])
            
            # Concatenate embeddings
            asset_emb = torch.cat([price_emb, tech_emb, fund_emb, market_emb], dim=-1)
            embeddings.append(asset_emb)
        
        # Combine all asset embeddings
        # [batch_size, seq_len, num_assets * hidden_dim]
        combined_emb = torch.cat(embeddings, dim=-1)
        
        # Project to fixed hidden dimension
        x = self.output_projection(combined_emb)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer encoder layers
        all_attentions = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x)
            all_attentions.append(attn_weights)
        
        return x, all_attentions

class AssetAttentionTransformer(nn.Module):
    """
    Transformer model with cross-asset attention mechanism.
    """
    def __init__(self, feature_dim, hidden_dim, num_heads, num_layers, num_assets, dropout=0.1, max_seq_len=60):
        """
        Initialize asset attention transformer.
        
        Args:
            feature_dim (int): Dimension of input features per asset
            hidden_dim (int): Hidden dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            num_assets (int): Number of assets
            dropout (float): Dropout rate
            max_seq_len (int): Maximum sequence length
        """
        super(AssetAttentionTransformer, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_assets = num_assets
        
        # Input embedding for each asset
        self.asset_embeddings = nn.ModuleList([
            nn.Linear(feature_dim, hidden_dim) for _ in range(num_assets)
        ])
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Temporal self-attention layers (process each asset's time series)
        self.temporal_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers // 2)  # Half the layers for temporal attention
        ])
        
        # Cross-asset attention layers
        self.asset_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers - num_layers // 2)  # Remaining layers for asset attention
        ])
        
        # Final output layer
        self.output_layer = nn.Linear(hidden_dim * num_assets, hidden_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized AssetAttentionTransformer with {num_layers} layers, "
                   f"{num_heads} heads, and hidden dimension {hidden_dim}")
        
    def _init_weights(self, module):
        """
        Initialize weights of the model.
        
        Args:
            module (nn.Module): Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_assets, feature_dim]
            
        Returns:
            torch.Tensor: Encoded market state
        """
        batch_size, seq_len, num_assets, _ = x.size()
        
        # Process each asset separately
        asset_embeddings = []
        temporal_attentions = []
        
        for i in range(num_assets):
            # Get features for this asset
            asset_features = x[:, :, i, :]
            
            # Apply asset-specific embedding
            asset_emb = self.asset_embeddings[i](asset_features)
            
            # Add positional encoding
            asset_emb = self.pos_encoding(asset_emb)
            
            # Apply temporal self-attention
            asset_temporal = asset_emb
            for layer in self.temporal_layers:
                asset_temporal, attn_weights = layer(asset_temporal)
                temporal_attentions.append(attn_weights)
            
            asset_embeddings.append(asset_temporal)
        
        # Concatenate all asset embeddings along feature dimension
        # Shape: [batch_size, seq_len, num_assets * hidden_dim]
        combined_emb = torch.cat(asset_embeddings, dim=-1)
        
        # Project to fixed hidden dimension
        # Shape: [batch_size, seq_len, hidden_dim]
        combined_emb = self.output_layer(combined_emb)
        
        # Apply cross-asset attention
        cross_asset_attentions = []
        cross_asset_emb = combined_emb
        
        for layer in self.asset_layers:
            cross_asset_emb, attn_weights = layer(cross_asset_emb)
            cross_asset_attentions.append(attn_weights)
        
        return cross_asset_emb, (temporal_attentions, cross_asset_attentions)