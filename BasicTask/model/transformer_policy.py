import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=10000):
		super(PositionalEncoding, self).__init__()
		
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		
		self.register_buffer('pe', pe)
	
	def forward(self, x):
		x = x + self.pe[:x.size(1), :]
		return x


class TransformerPolicy(nn.Module):
	
	def __init__(self, obs_dim: int, action_dim: int, d_model: int = 128, 
				 nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512, 
				 dropout: float = 0.1, max_seq_len: int = 10000, num_classes: int = 3,
				 obs_embed_hidden: int = 256, obs_embed_layers: int = 2):
		super(TransformerPolicy, self).__init__()
		
		self.obs_dim = obs_dim
		self.action_dim = action_dim
		self.d_model = d_model
		self.max_seq_len = max_seq_len
		self.num_classes = num_classes
		self.obs_embed_hidden = obs_embed_hidden
		self.obs_embed_layers = obs_embed_layers
		
		obs_embed_layers_list = []
		if obs_embed_layers == 1:
			obs_embed_layers_list.append(nn.Linear(obs_dim, d_model))
		elif obs_embed_layers == 2:
			obs_embed_layers_list.extend([
				nn.Linear(obs_dim, obs_embed_hidden),
				nn.GELU(),
				nn.LayerNorm(obs_embed_hidden),
				nn.Dropout(dropout),
				nn.Linear(obs_embed_hidden, d_model)
			])
		else:
			obs_embed_layers_list.append(nn.Linear(obs_dim, obs_embed_hidden))
			obs_embed_layers_list.append(nn.GELU())
			obs_embed_layers_list.append(nn.LayerNorm(obs_embed_hidden))
			obs_embed_layers_list.append(nn.Dropout(dropout))
			for _ in range(obs_embed_layers - 2):
				obs_embed_layers_list.append(nn.Linear(obs_embed_hidden, obs_embed_hidden))
				obs_embed_layers_list.append(nn.GELU())
				obs_embed_layers_list.append(nn.LayerNorm(obs_embed_hidden))
				obs_embed_layers_list.append(nn.Dropout(dropout))
			obs_embed_layers_list.append(nn.Linear(obs_embed_hidden, d_model))
		
		self.obs_embedding = nn.Sequential(*obs_embed_layers_list)
		self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
		
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True
		)
		self.transformer_encoder = nn.TransformerEncoder(
			encoder_layer,
			num_layers=num_layers
		)
		
		self.action_head = nn.Linear(d_model, action_dim * num_classes)
		
		self._init_weights()
	
	def _init_weights(self):
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)
	
	def forward(self, obs_seq: torch.Tensor, 
				causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		batch_size, seq_len, _ = obs_seq.shape
		
		x = self.obs_embedding(obs_seq)
		x = self.pos_encoder(x)
		
		if causal_mask is None:
			causal_mask = self._generate_causal_mask(seq_len, obs_seq.device)
		
		x = self.transformer_encoder(x, mask=causal_mask)
		
		logits = self.action_head(x)
		
		logits = logits.view(batch_size, seq_len, self.action_dim, self.num_classes)
		
		return logits
	
	def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
		mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
		return mask
	
	def predict_action_classes(self, obs_history: torch.Tensor) -> torch.Tensor:
		"""Predict action class indices for single action"""
		was_training = self.training
		self.eval()
		with torch.no_grad():
			if obs_history.dim() == 2:
				obs_history = obs_history.unsqueeze(0)
			
			logits = self.forward(obs_history)
			class_indices = torch.argmax(logits[0, -1, :, :], dim=-1)
		
		if was_training:
			self.train()
		return class_indices
