import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import TransformerPolicy

class TrajectoryDataset(Dataset):
	
	def __init__(self, data_dir, max_seq_len=None, pad=True, normalize_observations=True, 
				 obs_mean=None, obs_std=None, easy=False):

		self.data_dir = data_dir
		self.trajectories = []
		self.max_seq_len = max_seq_len
		self.pad = pad
		self.normalize_observations = normalize_observations
		self.obs_mean = obs_mean
		self.obs_std = obs_std
		self.max_obs_dim = None
		self.easy = easy
		
		self._load_trajectories()
		self._compute_global_obs_dim()
		
		if max_seq_len is None and self.trajectories:
			self.max_seq_len = max(len(t['observations']) for t in self.trajectories)
		
		if self.normalize_observations:
			if self.obs_mean is None or self.obs_std is None:
				self._compute_obs_stats()
			else:
				print(f"\nUsing provided observation normalization statistics:")
				print(f"  Mean shape: {self.obs_mean.shape}")
				print(f"  Std shape: {self.obs_std.shape}")
		
		self._preprocess_data()
		self._analyze_data()
	
	def _load_trajectories(self):
		files = [f for f in os.listdir(self.data_dir) 
				if f.startswith('trajectory_') and f.endswith('.pkl')]
		files.sort()
		
		print(f"Found {len(files)} trajectory files")
		if self.easy:
			print("Easy mode enabled: using filtered observations and actions")
			print("Only keeping data before connection (obs[0] = 0)")
		
		for fn in files:
			filepath = os.path.join(self.data_dir, fn)
			with open(filepath, 'rb') as f:
				traj = pickle.load(f)
				
			obs_seq = np.array(traj['observations'], dtype=np.float32)
			action_seq = np.array(traj['actions'], dtype=np.float32)
			
			# Filter data if in easy mode
			if self.easy:
				obs_seq, action_seq = self._filter_easy_data(obs_seq, action_seq)
				
				# Skip trajectories that became too short after filtering
				if obs_seq is None:
					continue
			
			self.trajectories.append({
				'observations': obs_seq,
				'actions': action_seq,
				'seq_len': len(obs_seq)
			})
	
	def _compute_global_obs_dim(self):
		if not self.trajectories:
			self.max_obs_dim = 0
			return
		
		self.max_obs_dim = len(self.trajectories[0]['observations'][0])
		print(f"Observation dimension: {self.max_obs_dim}")
	
	def _compute_obs_stats(self):
		all_obs = np.concatenate([np.array(t['observations']) for t in self.trajectories], axis=0)
		self.obs_mean = np.mean(all_obs, axis=0, keepdims=True).astype(np.float32)
		self.obs_std = np.std(all_obs, axis=0, keepdims=True).astype(np.float32)
		self.obs_std = np.where(self.obs_std < 1e-6, 1.0, self.obs_std)
		print(f"\nObservation normalization:")
		print(f"  Mean shape: {self.obs_mean.shape}")
		print(f"  Std shape: {self.obs_std.shape}")
		print(f"  Mean range: [{self.obs_mean.min():.4f}, {self.obs_mean.max():.4f}]")
		print(f"  Std range: [{self.obs_std.min():.4f}, {self.obs_std.max():.4f}]")
	
	def _filter_easy_data(self, obs_seq, action_seq):
		"""Filter data for easy mode: keep arm state + AB rod states, xyz movement + gripper control
		Only keep data before connection happens (obs[:, 0] == 0)"""
		# Find the first timestep where connection happens (obs[0] == 1)
		# Only keep data before connection
		conn_indices = np.where(obs_seq[:, 0] >= 0.5)[0]  # obs[0] == 1 means connected
		
		if len(conn_indices) > 0:
			# Keep only data before first connection
			cutoff_idx = conn_indices[0]
			obs_seq = obs_seq[:cutoff_idx]
			action_seq = action_seq[:cutoff_idx]
		
		# Skip if trajectory is too short (after potential connection cutoff)
		if len(obs_seq) < 10:
			return None, None
		
		# Observations to keep:
		# - Indices 1-7: joint positions (7 values)
		# - Indices 8-14: joint velocities (7 values)
		# - Indices 15-17: end-effector position (3 values)
		# - Indices 18-21: end-effector orientation (4 values)
		# - Indices 22-34: rod A state (13 values)
		# - Indices 35-47: rod B state (13 values)
		# Total: 7 + 7 + 3 + 4 + 13 + 13 = 47 values
		obs_filtered = obs_seq[:, 1:48]
		
		# Actions to keep:
		# - Indices 0-2: xyz movement
		# - Index 6: gripper control
		action_filtered = np.concatenate([action_seq[:, 0:3], action_seq[:, 6:7]], axis=1)
		
		return obs_filtered, action_filtered
	
	def _preprocess_data(self):
		if not self.trajectories:
			return
		
		processed_trajectories = []
		for traj in self.trajectories:
			obs_seq = traj['observations'].copy()
			action_seq = traj['actions'].copy()
			
			if self.normalize_observations:
				obs_seq = (obs_seq.astype(np.float32) - 
						  self.obs_mean.astype(np.float32)) / self.obs_std.astype(np.float32)
			
			eps = 1e-6
			action_classes = np.where(action_seq > eps, 2, np.where(action_seq < -eps, 0, 1))
			
			if self.pad and self.max_seq_len:
				seq_len = len(obs_seq)
				if seq_len < self.max_seq_len:
					pad_len = self.max_seq_len - seq_len
					obs_seq = np.pad(obs_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
					action_classes = np.pad(action_classes, ((0, pad_len), (0, 0)), mode='constant', constant_values=-100)
				elif seq_len > self.max_seq_len:
					obs_seq = obs_seq[:self.max_seq_len]
					action_classes = action_classes[:self.max_seq_len]
			
			processed_trajectories.append({
				'observations': obs_seq,
				'actions': action_classes,
				'seq_len': traj['seq_len']
			})
		
		self.trajectories = processed_trajectories
		print(f"\nPreprocessing completed: {len(self.trajectories)} trajectories")
	
	def _analyze_data(self):
		if not self.trajectories:
			return
		
		lengths = [t['seq_len'] for t in self.trajectories]
		
		print(f"\nDataset Statistics:")
		print(f"  Total trajectories: {len(self.trajectories)}")
		print(f"  Original sequence length - min: {min(lengths)}, max: {max(lengths)}, "
			  f"mean: {np.mean(lengths):.1f}, std: {np.std(lengths):.1f}")
		if self.pad and self.max_seq_len:
			print(f"  Padded to max_seq_len: {self.max_seq_len}")
		print(f"  Observation dimension: {self.max_obs_dim}")
		
		# Analyze action class distribution
		all_actions = np.concatenate([t['actions'] for t in self.trajectories], axis=0)
		action_dim = all_actions.shape[-1]
		
		print(f"\nAction class distribution (excluding padding):")
		for dim in range(action_dim):
			valid_mask = all_actions[:, dim] != -100
			dim_actions = all_actions[valid_mask, dim]
			class_counts = np.bincount(dim_actions, minlength=3)
			total = len(dim_actions)
			
			print(f"  Dimension {dim}:")
			for cls in range(3):
				count = class_counts[cls]
				percentage = count / total * 100 if total > 0 else 0
				print(f"    Class {cls}: {count:6d} ({percentage:5.2f}%)")
	
	def __len__(self):
		return len(self.trajectories)
	
	def __getitem__(self, idx):
		traj = self.trajectories[idx]
		
		return {
			'observations': torch.from_numpy(traj['observations']),
			'actions': torch.from_numpy(traj['actions']),
			'seq_len': traj['seq_len']
		}


def collate_fn(batch):
	obs_seq = torch.stack([item['observations'] for item in batch])
	action_seq = torch.stack([item['actions'] for item in batch])
	seq_lens = torch.tensor([item['seq_len'] for item in batch])
	
	return {
		'observations': obs_seq,
		'actions': action_seq,
		'seq_lens': seq_lens
	}


def compute_class_weights(dataset):
	"""Compute inverse frequency class weights to handle class imbalance"""
	all_actions = np.concatenate([t['actions'] for t in dataset.trajectories], axis=0)
	valid_mask = all_actions != -100
	
	class_counts = np.bincount(all_actions[valid_mask].flatten(), minlength=3)
	total_samples = np.sum(class_counts)
	
	# Compute inverse frequency weights
	# Add small epsilon to avoid division by zero
	class_weights = 1.0 / (class_counts / total_samples + 1e-6)
	
	# Normalize weights
	class_weights = class_weights / np.sum(class_weights) * 3
	
	print(f"\nClass weights (inverse frequency):")
	for cls in range(3):
		print(f"  Class {cls}: {class_weights[cls]:.4f} (count: {class_counts[cls]})")
	
	return torch.FloatTensor(class_weights).cuda() if torch.cuda.is_available() else torch.FloatTensor(class_weights)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs, class_weights=None, entropy_weight=0.01):
	model.train()
	total_loss = 0
	total_cls_loss = 0
	total_entropy_loss = 0
	
	pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
	
	for batch_idx, batch in enumerate(pbar):
		obs_seq = batch['observations'].to(device)
		action_classes = batch['actions'].to(device)
		seq_lens = batch['seq_lens'].to(device)
		
		logits = model(obs_seq)
		
		batch_size, seq_len = action_classes.shape[:2]
		positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
		mask = positions < seq_lens.unsqueeze(1)
		
		# logits: (batch_size, seq_len, action_dim, num_classes)
		# action_classes: (batch_size, seq_len, action_dim)
		
		# Flatten all dimensions except the last one (num_classes)
		num_classes = logits.shape[-1]
		logits_flat = logits.view(-1, num_classes)  # (batch_size * seq_len * action_dim, num_classes)
		action_classes_flat = action_classes.view(-1)  # (batch_size * seq_len * action_dim,)
		
		# Create mask for valid positions
		mask_expanded = mask.unsqueeze(-1).expand(-1, -1, action_classes.shape[-1])  # (batch_size, seq_len, action_dim)
		mask_flat = mask_expanded.reshape(-1)  # (batch_size * seq_len * action_dim,)
		
		if class_weights is not None:
			# Apply class weights to the loss
			# Create weight matrix for each sample
			weights = torch.ones_like(action_classes_flat, dtype=torch.float)
			valid_mask = action_classes_flat != -100
			weights[valid_mask] = class_weights[action_classes_flat[valid_mask]]
			loss_f = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100, reduction='none')
			cls_loss_per_sample = loss_f(logits_flat, action_classes_flat)
			cls_loss = (cls_loss_per_sample * weights).sum() / weights.sum()
		else:
			cls_loss = criterion(logits_flat, action_classes_flat)
		
		# Compute entropy regularization loss to encourage exploration
		# Entropy = -sum(p * log(p)) where p is softmax probability
		if entropy_weight > 0:
			probs = torch.softmax(logits_flat, dim=-1)
			log_probs = torch.log_softmax(logits_flat, dim=-1)
			entropy = -(probs * log_probs).sum(dim=-1)
			# Only compute entropy for valid positions
			entropy_loss = entropy[mask_flat].mean()
		else:
			entropy_loss = torch.tensor(0.0, device=device)
		
		# Total loss = classification loss + entropy regularization
		loss = cls_loss + entropy_weight * entropy_loss
		
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		
		total_loss += loss.item()
		total_cls_loss += cls_loss.item()
		total_entropy_loss += entropy_loss.item()
		
		pbar.set_postfix({
			'loss': f'{loss.item():.4f}',
			'cls': f'{cls_loss.item():.4f}',
			'ent': f'{entropy_loss.item():.4f}'
		})
	
	avg_loss = total_loss / len(dataloader)
	avg_cls_loss = total_cls_loss / len(dataloader)
	avg_entropy_loss = total_entropy_loss / len(dataloader)
	
	return avg_loss, avg_cls_loss, avg_entropy_loss


def validate(model, dataloader, criterion, device, class_weights=None):
	model.eval()
	total_loss = 0
	
	with torch.no_grad():
		for batch in dataloader:
			obs_seq = batch['observations'].to(device)
			action_classes = batch['actions'].to(device)
			seq_lens = batch['seq_lens'].to(device)
			
			logits = model(obs_seq)
			
			batch_size, seq_len = action_classes.shape[:2]
			positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
			mask = positions < seq_lens.unsqueeze(1)
			
			# logits: (batch_size, seq_len, action_dim, num_classes)
			# action_classes: (batch_size, seq_len, action_dim)
			
			# Flatten all dimensions except the last one (num_classes)
			num_classes = logits.shape[-1]
			logits_flat = logits.view(-1, num_classes)  # (batch_size * seq_len * action_dim, num_classes)
			action_classes_flat = action_classes.view(-1)  # (batch_size * seq_len * action_dim,)
			
			# Create mask for valid positions
			mask_expanded = mask.unsqueeze(-1).expand(-1, -1, action_classes.shape[-1])  # (batch_size, seq_len, action_dim)
			mask_flat = mask_expanded.reshape(-1)  # (batch_size * seq_len * action_dim,)
			
			if class_weights is not None:
				# Apply class weights to loss (same as in train_epoch)
				# Create weight matrix for each sample
				weights = torch.ones_like(action_classes_flat, dtype=torch.float)
				valid_mask = action_classes_flat != -100
				weights[valid_mask] = class_weights[action_classes_flat[valid_mask]]
				loss_f = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100, reduction='none')
				cls_loss_per_sample = loss_f(logits_flat, action_classes_flat)
				loss = (cls_loss_per_sample * weights).sum() / weights.sum()
			else:
				loss = criterion(logits_flat, action_classes_flat)
			
			total_loss += loss.item()
	
	avg_loss = total_loss / len(dataloader)
	
	return avg_loss


def plot_training_curves(train_losses, val_losses, cls_losses, entropy_losses, save_path):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
	
	# Total loss
	ax1.plot(train_losses, label='Train Loss', linewidth=2)
	ax1.plot(val_losses, label='Val Loss', linewidth=2)
	ax1.set_xlabel('Epoch', fontsize=12)
	ax1.set_ylabel('Loss', fontsize=12)
	ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
	ax1.legend(fontsize=11)
	ax1.grid(True, alpha=0.3)
	
	# Component losses
	ax2.plot(cls_losses, label='Classification Loss', linewidth=2)
	ax2.plot(entropy_losses, label='Entropy Loss', linewidth=2)
	ax2.set_xlabel('Epoch', fontsize=12)
	ax2.set_ylabel('Loss', fontsize=12)
	ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
	ax2.legend(fontsize=11)
	ax2.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	print(f"Training curves saved to {save_path}")
	plt.close()


def main():
	parser = argparse.ArgumentParser(description='Train Transformer policy network')
	parser.add_argument('--data-dir', type=str, default='data', 
					   help='Base data directory path (training data in data/train, validation data in data/val)')
	parser.add_argument('--train-dir', type=str, default=None, 
					   help='Training data directory path (overrides --data-dir)')
	parser.add_argument('--val-dir', type=str, default=None, 
					   help='Validation data directory path (overrides --data-dir)')
	parser.add_argument('--epochs', type=int, default=100, 
					   help='Number of training epochs')
	parser.add_argument('--batch-size', type=int, default=8, 
					   help='Batch size')
	parser.add_argument('--lr', type=float, default=1e-3, 
					   help='Learning rate')
	parser.add_argument('--d-model', type=int, default=128, 
					   help='Transformer model dimension')
	parser.add_argument('--nhead', type=int, default=8, 
					   help='Number of attention heads')
	parser.add_argument('--num-layers', type=int, default=4, 
					   help='Number of Transformer layers')
	parser.add_argument('--dim-feedforward', type=int, default=512, 
					   help='Feedforward network dimension')
	parser.add_argument('--dropout', type=float, default=0.1, 
					   help='Dropout rate')
	parser.add_argument('--max-seq-len', type=int, default=5000, 
					   help='Maximum sequence length, None means auto-determine')
	parser.add_argument('--no-pad', action='store_true', 
					   help='Disable padding')
	parser.add_argument('--seed', type=int, default=42, 
					   help='Random seed')
	parser.add_argument('--save-dir', type=str, default='checkpoints', 
					   help='Model save directory')
	parser.add_argument('--device', type=str, default='cuda', 
					   help='Device (cuda/cpu)')
	parser.add_argument('--early-stopping-patience', type=int, default=15,
					   help='Early stopping patience (number of epochs without improvement)')
	parser.add_argument('--early-stopping-delta', type=float, default=1e-6,
					   help='Minimum change to qualify as an improvement (default: 1e-6)')
	parser.add_argument('--pos-speed', type=float, default=0.5,
					   help='Position speed for discrete actions (default: 0.5)')
	parser.add_argument('--rot-speed', type=float, default=0.5,
					   help='Rotation speed for discrete actions (default: 0.5)')
	parser.add_argument('--obs-embed-hidden', type=int, default=256,
					   help='Hidden dimension for observation embedding MLP (default: 256)')
	parser.add_argument('--obs-embed-layers', type=int, default=2,
					   help='Number of layers in observation embedding MLP (default: 2)')
	parser.add_argument('--use-class-weights', action='store_true',
					   help='Use class weights to handle class imbalance')
	parser.add_argument('--entropy-weight', type=float, default=0.01,
					   help='Entropy regularization weight (default: 0.01). Set to 0 to disable.')
	parser.add_argument('--easy', action='store_true',
					   help='Easy mode: only use arm state + AB rod states, xyz movement + gripper control')
	
	args = parser.parse_args()
	
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	os.makedirs(args.save_dir, exist_ok=True)
	
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	
	if args.train_dir is None:
		train_dir = os.path.join(args.data_dir, 'train')
	else:
		train_dir = args.train_dir
	
	if args.val_dir is None:
		val_dir = os.path.join(args.data_dir, 'val')
	else:
		val_dir = args.val_dir
	
	print("\n" + "="*60)
	print("Loading data...")
	print("="*60)
	
	if not os.path.exists(train_dir):
		print(f"Training data directory not found: {train_dir}")
		print("Please collect training data first using: python collect_data.py -t train")
		return
	
	if not os.path.exists(val_dir):
		print(f"Validation data directory not found: {val_dir}")
		print("Please collect validation data first using: python collect_data.py -t val")
		return
	
	print(f"Loading training data from: {train_dir}")
	train_dataset = TrajectoryDataset(
		train_dir, 
		pad=not args.no_pad,
		normalize_observations=True,
		obs_mean=None,
		obs_std=None,
		easy=args.easy
	)
	
	print(f"\nLoading validation data from: {val_dir}")
	val_dataset = TrajectoryDataset(
		val_dir, 
		pad=not args.no_pad,
		normalize_observations=True,
		obs_mean=train_dataset.obs_mean,
		obs_std=train_dataset.obs_std,
		easy=args.easy
	)
	
	if len(train_dataset) == 0:
		print("No training data found!")
		return
	
	if len(val_dataset) == 0:
		print("No validation data found!")
		return
	
	print(f"\nDataset split:")
	print(f"  Train: {len(train_dataset)} trajectories")
	print(f"  Validation: {len(val_dataset)} trajectories")
	
	train_loader = DataLoader(
		train_dataset, 
		batch_size=args.batch_size, 
		shuffle=True, 
		collate_fn=collate_fn,
		num_workers=2,
		pin_memory=True if device.type == 'cuda' else False
	)
	
	val_loader = DataLoader(
		val_dataset, 
		batch_size=args.batch_size, 
		shuffle=False, 
		collate_fn=collate_fn,
		num_workers=2,
		pin_memory=True if device.type == 'cuda' else False
	)
	
	obs_dim = train_dataset.max_obs_dim
	
	sample_action = train_dataset[0]['actions']
	action_dim = sample_action.shape[-1]
	max_seq_len = train_dataset.max_seq_len
	
	print(f"\nModel configuration:")
	print(f"  Observation dimension: {obs_dim}")
	print(f"  Action dimension: {action_dim}")
	print(f"  Max sequence length: {max_seq_len}")
	print(f"  d_model: {args.d_model}")
	print(f"  nhead: {args.nhead}")
	print(f"  num_layers: {args.num_layers}")
	print(f"  dim_feedforward: {args.dim_feedforward}")
	print(f"  dropout: {args.dropout}")
	
	print("\n" + "="*60)
	print("Creating model...")
	print("="*60)
	
	model = TransformerPolicy(
		obs_dim=obs_dim,
		action_dim=action_dim,
		d_model=args.d_model,
		nhead=args.nhead,
		num_layers=args.num_layers,
		dim_feedforward=args.dim_feedforward,
		dropout=args.dropout,
		max_seq_len=max_seq_len,
		obs_embed_hidden=args.obs_embed_hidden,
		obs_embed_layers=args.obs_embed_layers
	).to(device)
	
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"\nModel parameters:")
	print(f"  Total: {total_params:,}")
	print(f"  Trainable: {trainable_params:,}")
	
	# Compute class weights if requested
	class_weights = None
	if args.use_class_weights:
		print("\n" + "="*60)
		print("Computing class weights...")
		print("="*60)
		class_weights = compute_class_weights(train_dataset)
	
	# Cross-Entropy Loss with ignore_index for padding
	criterion = nn.CrossEntropyLoss(ignore_index=-100)
	optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
	scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
	
	print("\n" + "="*60)
	print("Starting training...")
	print("="*60)
	print(f"\nTraining configuration:")
	print(f"  Use class weights: {args.use_class_weights}")
	print(f"  Entropy regularization weight: {args.entropy_weight}")
	
	train_losses = []
	val_losses = []
	cls_losses = []
	entropy_losses = []
	best_val_loss = float('inf')
	patience_counter = 0
	best_epoch = 0
	
	print(f"\nEarly stopping configuration:")
	print(f"  Patience: {args.early_stopping_patience} epochs")
	print(f"  Minimum improvement: {args.early_stopping_delta}")
	
	for epoch in range(1, args.epochs + 1):
		print(f"\nEpoch {epoch}/{args.epochs}")
		print("-" * 60)
		
		train_loss, cls_loss, entropy_loss = train_epoch(
			model, train_loader, optimizer, criterion, device, epoch, args.epochs,
			class_weights=class_weights, entropy_weight=args.entropy_weight
		)
		train_losses.append(train_loss)
		cls_losses.append(cls_loss)
		entropy_losses.append(entropy_loss)
		
		val_loss = validate(model, val_loader, criterion, device, class_weights=class_weights)
		val_losses.append(val_loss)
		
		scheduler.step()
		current_lr = optimizer.param_groups[0]['lr']
		
		print(f"\nEpoch {epoch}/{args.epochs} Summary:")
		print(f"  Train Loss: {train_loss:.6f} (cls: {cls_loss:.6f}, ent: {entropy_loss:.6f})")
		print(f"  Val Loss: {val_loss:.6f}")
		print(f"  Learning Rate: {current_lr:.6f}")
		
		if val_loss < best_val_loss - args.early_stopping_delta:
			best_val_loss = val_loss
			best_epoch = epoch
			patience_counter = 0
			
			best_model_path = os.path.join(args.save_dir, 'best_model.pth')
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler_state_dict': scheduler.state_dict(),
				'train_loss': train_loss,
				'val_loss': val_loss,
				'args': vars(args),
				'obs_dim': obs_dim,
				'action_dim': action_dim,
				'max_seq_len': max_seq_len,
				'obs_mean': train_dataset.obs_mean.tolist() if train_dataset.obs_mean is not None else None,
				'obs_std': train_dataset.obs_std.tolist() if train_dataset.obs_std is not None else None,
				'class_weights': class_weights.tolist() if class_weights is not None else None
			}, best_model_path)
			print(f"  ✓ Best model saved (val_loss: {best_val_loss:.6f})")
		else:
			patience_counter += 1
			print(f"  ℹ No improvement for {patience_counter} epoch(s) (best: {best_val_loss:.6f} at epoch {best_epoch})")
			
			if patience_counter >= args.early_stopping_patience:
				print(f"\n{'='*60}")
				print(f"Early stopping triggered!")
				print(f"  No improvement for {patience_counter} epochs")
				print(f"  Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
				print(f"{'='*60}")
				break
	
	final_model_path = os.path.join(args.save_dir, 'final_model.pth')
	torch.save({
		'epoch': args.epochs,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict': scheduler.state_dict(),
		'train_loss': train_losses[-1],
		'val_loss': val_losses[-1],
		'args': vars(args),
		'obs_dim': obs_dim,
		'action_dim': action_dim,
		'max_seq_len': max_seq_len,
		'obs_mean': train_dataset.obs_mean.tolist() if train_dataset.obs_mean is not None else None,
		'obs_std': train_dataset.obs_std.tolist() if train_dataset.obs_std is not None else None,
		'class_weights': class_weights.tolist() if class_weights is not None else None
	}, final_model_path)
	print(f"\nFinal model saved: {final_model_path}")
	
	plot_path = os.path.join(args.save_dir, 'training_curves.png')
	plot_training_curves(train_losses, val_losses, cls_losses, entropy_losses, plot_path)
	
	print("\n" + "="*60)
	print("Training completed!")
	print("="*60)
	print(f"Best validation loss: {best_val_loss:.6f}")
	print(f"Final train loss: {train_losses[-1]:.6f}")
	print(f"Final validation loss: {val_losses[-1]:.6f}")
	print(f"All models saved in: {args.save_dir}")


if __name__ == "__main__":
	main()
