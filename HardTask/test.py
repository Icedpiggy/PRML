#!/usr/bin/env python3
"""
Test script for trained Transformer policy network in randomized ArmEnv
"""

import os
import argparse
import pickle
import numpy as np
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import TransformerPolicy
from envs import ArmEnv


class PolicyTester:
	"""Tester for policy in randomized environment"""
	
	def __init__(self, model, device, pos_speed, rot_speed,
				 obs_mean, obs_std, obs_dim, action_dim, 
				 max_seq_len, debug=False, show_boundary=False, 
				 speed=1.0, temperature=1.0, no_op_threshold=50, easy=False,
				 step_by_step=False):
		self.model = model
		self.device = device
		self.pos_speed = pos_speed
		self.rot_speed = rot_speed
		self.obs_mean = np.array(obs_mean)
		self.obs_std = np.array(obs_std)
		self.obs_dim = obs_dim
		self.action_dim = action_dim
		self.max_seq_len = max_seq_len
		self.debug = debug
		self.show_boundary = show_boundary
		self.speed = speed
		self.temperature = temperature
		self.no_op_threshold = no_op_threshold
		self.easy = easy
		self.step_by_step = step_by_step
		
		self.model.eval()
		
		# History buffer for sequence model
		self.obs_history = []
		
		# Track consecutive no-op actions
		self.consecutive_no_ops = 0
		
		# Check if interactive input is available
		import sys
		self.input_available = sys.stdin.isatty()
		
		print(f"\nPolicy Tester initialized")
		print(f"  Debug mode: {debug}")
		print(f"  Show boundary: {show_boundary}")
		print(f"  Easy mode: {easy}")
		print(f"  Playback speed: {speed}x")
		print(f"  Position speed: {pos_speed}")
		print(f"  Rotation speed: {rot_speed}")
		print(f"  Temperature: {temperature}")
		print(f"  No-op threshold: {no_op_threshold} steps")
		print(f"  Step-by-step mode: {step_by_step}")
		
		if step_by_step and not self.input_available:
			print(f"\n[WARNING] Step-by-step mode requested but interactive input not available!")
			print(f"[WARNING] Running in continuous mode instead.")
			self.step_by_step = False
	
	def get_action(self, obs):
		"""Get action from model given current observation"""
		# Normalize observation
		obs_normalized = (obs.astype(np.float32) - 
						 self.obs_mean.astype(np.float32)) / self.obs_std.astype(np.float32)
		
		# Convert to tensor
		obs_tensor = torch.from_numpy(obs_normalized).float().unsqueeze(0).to(self.device)
		
		# Update history
		self.obs_history.append(obs_tensor)
		if len(self.obs_history) > self.max_seq_len:
			self.obs_history.pop(0)
		
		# Stack observations
		obs_seq = torch.cat(self.obs_history, dim=1)
		
		# Get action logits from model
		with torch.no_grad():
			action_logits = self.model(obs_seq)  # (1, seq_len, action_dim, num_classes)
		
		# Get logits for the last timestep
		logits = action_logits[0, -1, :, :]  # (action_dim, num_classes)
		
		# Apply temperature scaling to logits
		if self.temperature != 1.0:
			logits = logits / self.temperature
		
		# Get class indices (argmax on logits, same as on probs)
		class_indices = torch.argmax(logits, dim=-1).cpu().numpy()  # (action_dim,)
		
		# Check if all actions are no-op (all class 1)
		is_no_op = np.all(class_indices == 1)
		if is_no_op:
			self.consecutive_no_ops += 1
		else:
			self.consecutive_no_ops = 0
		
		# If too many consecutive no-ops, force exploration by randomly changing some actions
		if self.consecutive_no_ops >= self.no_op_threshold:
			if self.debug:
				print(f"\n[WARNING] {self.consecutive_no_ops} consecutive no-ops detected! Forcing exploration...")
			# Randomly pick 2-3 dimensions to change
			num_dims_to_change = np.random.randint(2, min(4, self.action_dim))
			dims_to_change = np.random.choice(self.action_dim, num_dims_to_change, replace=False)
			for dim in dims_to_change:
				# Randomly choose class 0 or 2 (not 1)
				class_indices[dim] = np.random.choice([0, 2])
			self.consecutive_no_ops = 0
		
		# Convert class indices back to continuous action values
		if self.easy:
			# Easy mode: only 4 action dimensions (xyz + gripper)
			action = np.zeros(7, dtype=np.float32)  # Full action space for environment
			for i in range(4):  # Only process first 4 dimensions
				if i == 3:  # Gripper dimension (index 3 in easy mode, maps to index 6 in full action)
					# Class 0 -> -1.0 (close), Class 1 -> 0, Class 2 -> 1.0 (open)
					action[6] = class_indices[i] - 1.0
				else:  # Position dimensions (0, 1, 2)
					# Class 0 -> -pos_speed, Class 1 -> 0, Class 2 -> +pos_speed
					action[i] = (class_indices[i] - 1) * self.pos_speed
		else:
			# Normal mode: 7 action dimensions
			action = np.zeros(7, dtype=np.float32)
			for i in range(7):
				if i == 6:  # Gripper dimension
					# Class 0 -> -1.0 (close), Class 1 -> 0, Class 2 -> 1.0 (open)
					action[i] = class_indices[i] - 1.0
				elif i < 3:  # Position dimensions (0, 1, 2)
					# Class 0 -> -pos_speed, Class 1 -> 0, Class 2 -> +pos_speed
					action[i] = (class_indices[i] - 1) * self.pos_speed
				else:  # Rotation dimensions (3, 4, 5)
					# Class 0 -> -rot_speed, Class 1 -> 0, Class 2 -> +rot_speed
					action[i] = (class_indices[i] - 1) * self.rot_speed
		
		# Store debug info for later display
		self.last_debug_info = {
			'seq_len': obs_seq.shape[1],
			'logits_range': (float(logits.min()), float(logits.max())),
			'class_indices': class_indices,
			'action': action,
			'action_norm': float(np.linalg.norm(action)),
			'pos_delta': action[:3],
			'rot_delta': action[3:6],
			'gripper_cmd': action[6],
			'consecutive_no_ops': self.consecutive_no_ops,
			'temperature': self.temperature,
			'easy_mode': self.easy
		}
		
		return action
	
	def reset_history(self):
		"""Reset observation history"""
		self.obs_history = []
	
	def test_episode(self, env, max_steps=5000, view='front', render=True):
		"""Test model on one episode"""
		self.reset_history()
		
		if render:
			env.set_camera_view(view)
		obs = env.reset()
		
		done = False
		step = 0
		info_history = []
		
		if self.step_by_step:
			print("\n" + "="*80)
			print("STEP-BY-STEP MODE ENABLED")
			print("="*80)
			print("  Press Enter after each step to continue")
			print("  Type 'skip' to run continuously")
			print("  Type 'q' to quit episode")
			print("="*80)
		
		try:
			# Create progress bar
			pbar = tqdm(range(max_steps), desc="Episode", disable=self.step_by_step or not self.debug,
						unit="step")
			
			for step in pbar:
				action = self.get_action(obs)
				
				if self.debug or self.step_by_step:
					print(f"\n{'-'*80}")
					print(f"Step {step}")
					print(f"{'-'*80}")
					if self.debug:
						debug_info = self.last_debug_info
						print(f"\n=== ACTION ===")
						print(f"Position delta: ({debug_info['pos_delta'][0]:.4f}, {debug_info['pos_delta'][1]:.4f}, {debug_info['pos_delta'][2]:.4f})")
						if not self.easy:
							print(f"Rotation delta: ({debug_info['rot_delta'][0]:.4f}, {debug_info['rot_delta'][1]:.4f}, {debug_info['rot_delta'][2]:.4f})")
						print(f"Gripper command: {debug_info['gripper_cmd']:.2f}")
						print(f"Action norm: {debug_info['action_norm']:.4f}")
						print(f"\n=== MODEL OUTPUT ===")
						print(f"Sequence length: {debug_info['seq_len']}")
						print(f"Logits range: [{debug_info['logits_range'][0]:.2f}, {debug_info['logits_range'][1]:.2f}]")
						print(f"Temperature: {debug_info['temperature']:.2f}")
						print(f"Consecutive no-ops: {debug_info['consecutive_no_ops']}")
						print(f"Class indices: {debug_info['class_indices']}")
				
				done, info = env.step(action)
				obs = env.get_obs()
				
				info_history.append(info.copy())
				
				# Update progress bar with debug info
				if self.debug and not self.step_by_step:
					debug_info = self.last_debug_info
					
					# Format debug info for progress bar
					postfix = []
					postfix.append(f"conn={info['conn']}")
					postfix.append(f"hit={info['hit']}")
					
					if env.conn and env.comb:
						ends = env._get_ends(env.comb, env.COMB_L)
						d = min(np.linalg.norm(e - np.array(env.tgt_pos)) for e in ends)
						postfix.append(f"dist={d:.3f}m")
					
					# Add action info
					postfix.append(f"pos=({debug_info['pos_delta'][0]:.3f},{debug_info['pos_delta'][1]:.3f},{debug_info['pos_delta'][2]:.3f})")
					if not self.easy:
						postfix.append(f"rot=({debug_info['rot_delta'][0]:.3f},{debug_info['rot_delta'][1]:.3f},{debug_info['rot_delta'][2]:.3f})")
					postfix.append(f"grip={debug_info['gripper_cmd']:.1f}")
					
					# Add model output info
					postfix.append(f"seq_len={debug_info['seq_len']}")
					postfix.append(f"logits=[{debug_info['logits_range'][0]:.2f},{debug_info['logits_range'][1]:.2f}]")
					postfix.append(f"temp={debug_info['temperature']:.2f}")
					postfix.append(f"no_op={debug_info['consecutive_no_ops']}")
					
					# Format class indices
					classes_str = ''.join(map(str, debug_info['class_indices']))
					postfix.append(f"classes={classes_str}")
					
					pbar.set_postfix_str(' '.join(postfix))
				
				if self.debug or self.step_by_step:
					print(f"\n=== ENVIRONMENT INFO ===")
					print(f"Connected: {info['conn']}")
					print(f"Hit target: {info['hit']}")
					print(f"Boundary violation: {info['bnd_vio']}")
					print(f"Episode done: {done}")
				
				if self.step_by_step and not done:
					print(f"\n{'='*80}")
					try:
						user_input = input("Press Enter to continue (or type 'skip'/'q'): ")
						if user_input.strip().lower() == 'skip':
							self.step_by_step = False
							print("\n[INFO] Switching to continuous mode...")
						elif user_input.strip().lower() == 'q':
							print("\n[INFO] Quitting episode...")
							break
					except (EOFError, KeyboardInterrupt):
						print("\n[ERROR] Input not available!")
						print("[ERROR] Switching to continuous mode...")
						self.step_by_step = False
					except Exception as e:
						print(f"\n[ERROR] Unexpected error getting input: {e}")
						print("[ERROR] Switching to continuous mode...")
						self.step_by_step = False
				
				if done:
					if self.debug or self.step_by_step:
						print(f"\n{'-'*80}")
						print(f"Episode finished at step {step}")
						print(f"Reason: {'Target hit' if info['hit'] else 'Boundary violation'}")
						print(f"{'-'*80}")
					break
				
				if render:
					time.sleep(1./ (self.speed * 60))
			
			pbar.close()
			
			success = info['hit'] and not info['bnd_vio']
			
			return {
				'success': success,
				'steps': step + 1,
				'connected': info['conn'],
				'hit_target': info['hit'],
				'boundary_violation': info['bnd_vio'],
				'info_history': info_history
			}
		
		except Exception as e:
			print(f"\nError during episode: {e}")
			import traceback
			traceback.print_exc()
			
			return {
				'success': False,
				'steps': step + 1,
				'connected': False,
				'hit_target': False,
				'boundary_violation': True,
				'error': str(e),
				'info_history': info_history
			}


def test_model(env_config, model_path, device, num_episodes=10, max_steps=5000,
				view='front', debug=False, show_boundary=False, speed=1.0,
				temperature=1.0, no_op_threshold=50, easy=False, step_by_step=False,
				render=True, seed=None):
	"""Test model in randomized environment"""
	
	print("\n" + "="*60)
	print("Loading model...")
	print("="*60)
	
	if not os.path.exists(model_path):
		print(f"Model file not found: {model_path}")
		return None
	
	checkpoint = torch.load(model_path, map_location=device)
	
	obs_dim = checkpoint['obs_dim']
	action_dim = checkpoint['action_dim']
	max_seq_len = checkpoint['max_seq_len']
	
	# Get action speeds from checkpoint or use defaults
	pos_speed = checkpoint['args'].get('pos_speed', 0.5)
	rot_speed = checkpoint['args'].get('rot_speed', 0.5)
	
	obs_mean = np.array(checkpoint['obs_mean']) if checkpoint['obs_mean'] is not None else None
	obs_std = np.array(checkpoint['obs_std']) if checkpoint['obs_std'] is not None else None
	
	if obs_mean is None or obs_std is None:
		print("\nWarning: Observation normalization statistics not found in checkpoint!")
		print("Model was trained without observation normalization.")
		# Use identity normalization (mean=0, std=1) as numpy arrays
		obs_mean = np.zeros((1, obs_dim), dtype=np.float32)
		obs_std = np.ones((1, obs_dim), dtype=np.float32)
	
	print(f"  Observation dimension: {obs_dim}")
	print(f"  Action dimension: {action_dim}")
	print(f"  Max sequence length: {max_seq_len}")
	print(f"  Easy mode: {easy}")
	print(f"  Model epoch: {checkpoint.get('epoch', 'N/A')}")
	print(f"  Train loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
	print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
	print(f"  Position speed: {pos_speed}")
	print(f"  Rotation speed: {rot_speed}")
	
	if checkpoint.get('obs_mean') is not None:
		print(f"\nObservation normalization:")
		print(f"  Mean shape: {obs_mean.shape}")
		print(f"  Std shape: {obs_std.shape}")
	
	print("\n" + "="*60)
	print("Creating model...")
	print("="*60)
	
	# Get obs_embed parameters from checkpoint or use defaults
	obs_embed_hidden = checkpoint['args'].get('obs_embed_hidden', 256)
	obs_embed_layers = checkpoint['args'].get('obs_embed_layers', 2)
	
	model = TransformerPolicy(
		obs_dim=obs_dim,
		action_dim=action_dim,
		d_model=checkpoint['args']['d_model'],
		nhead=checkpoint['args']['nhead'],
		num_layers=checkpoint['args']['num_layers'],
		dim_feedforward=checkpoint['args']['dim_feedforward'],
		dropout=checkpoint['args']['dropout'],
		max_seq_len=max_seq_len,
		obs_embed_hidden=obs_embed_hidden,
		obs_embed_layers=obs_embed_layers
	).to(device)
	
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()
	
	total_params = sum(p.numel() for p in model.parameters())
	print(f"\nModel parameters: {total_params:,}")
	
	tester = PolicyTester(
		model, device, pos_speed, rot_speed, obs_mean, obs_std,
		obs_dim, action_dim, max_seq_len, debug=debug, 
		show_boundary=show_boundary, speed=speed,
		temperature=temperature, no_op_threshold=no_op_threshold, easy=easy,
		step_by_step=step_by_step
	)
	
	print("\n" + "="*60)
	print("Testing in randomized environment...")
	print("="*60)
	print(f"  Environment config: {env_config}")
	print(f"  Number of episodes: {num_episodes}")
	print(f"  Max steps per episode: {max_steps}")
	print(f"  View: {view}")
	print(f"  Render: {render}")
	print(f"  Seed: {seed if seed is not None else 'Random'}")
	if step_by_step:
		print(f"  Step-by-step mode: ENABLED")
	
	results = []
	
	for episode in range(num_episodes):
		print(f"\n{'='*60}")
		print(f"Episode {episode + 1}/{num_episodes}")
		print(f"{'='*60}")
		
		if seed is None:
			episode_seed = None
		else:
			episode_seed = seed + episode
		
		env = ArmEnv(render=render, verbose=False, debug=debug, 
					show_bnd=show_boundary, seed=episode_seed, **env_config)
		
		result = tester.test_episode(env, max_steps=max_steps, view=view, render=render)
		results.append(result)
		
		print(f"\nEpisode {episode + 1} Results:")
		print(f"  Success: {'✓' if result['success'] else '✗'}")
		print(f"  Steps: {result['steps']}")
		print(f"  Connected: {'✓' if result['connected'] else '✗'}")
		print(f"  Hit target: {'✓' if result['hit_target'] else '✗'}")
		print(f"  Boundary violation: {'✓' if result['boundary_violation'] else '✗'}")
		
		if 'error' in result:
			print(f"  Error: {result['error']}")
		
		env.close()
		
		if episode < num_episodes - 1 and render:
			input("\nPress Enter to continue to next episode (or Ctrl+C to exit)...")
	
	# Calculate statistics
	success_rate = np.mean([r['success'] for r in results])
	avg_steps = np.mean([r['steps'] for r in results])
	connect_rate = np.mean([r['connected'] for r in results])
	hit_rate = np.mean([r['hit_target'] for r in results])
	violation_rate = np.mean([r['boundary_violation'] for r in results])
	
	print("\n" + "="*60)
	print("Overall Results")
	print("="*60)
	print(f"  Success rate: {success_rate:.2%} ({sum(r['success'] for r in results)}/{len(results)})")
	print(f"  Average steps: {avg_steps:.1f}")
	print(f"  Connection rate: {connect_rate:.2%}")
	print(f"  Hit target rate: {hit_rate:.2%}")
	print(f"  Boundary violation rate: {violation_rate:.2%}")
	
	return {
		'success_rate': success_rate,
		'avg_steps': avg_steps,
		'connect_rate': connect_rate,
		'hit_rate': hit_rate,
		'violation_rate': violation_rate,
		'episode_results': results,
		'model_info': {
			'epoch': checkpoint.get('epoch'),
			'train_loss': checkpoint.get('train_loss'),
			'val_loss': checkpoint.get('val_loss'),
			'total_params': total_params
		}
	}


def plot_success_rates(results_by_mode, save_path):
	"""Plot success rates for different environment modes"""
	modes = list(results_by_mode.keys())
	success_rates = [r['success_rate'] for r in results_by_mode.values()]
	
	fig, ax = plt.subplots(1, 1, figsize=(10, 6))
	
	bars = ax.bar(range(len(modes)), success_rates, color='steelblue', alpha=0.7)
	ax.set_xlabel('Environment Mode', fontsize=12)
	ax.set_ylabel('Success Rate', fontsize=12)
	ax.set_title('Policy Performance in Different Environment Modes', fontsize=14, fontweight='bold')
	ax.set_xticks(range(len(modes)))
	ax.set_xticklabels(modes, rotation=0)
	ax.set_ylim([0, 1.05])
	ax.grid(True, alpha=0.3, axis='y')
	
	# Add percentage labels on bars
	for bar, rate in zip(bars, success_rates):
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width()/2., height,
				f'{rate:.1%}',
				ha='center', va='bottom', fontsize=11, fontweight='bold')
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	print(f"\nSuccess rate plot saved to {save_path}")
	plt.close()


def main():
	parser = argparse.ArgumentParser(
		description='Test trained Transformer policy in randomized ArmEnv',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Test model with default settings
  python test.py --model-path checkpoints/best_model.pth
  
  # Test easy mode model
  python test.py --model-path checkpoints/best_model.pth --easy
  
  # Test with 20 episodes in hard mode
  python test.py --model-path checkpoints/best_model.pth --episodes 20 --hard
  
  # Test with debug info and boundary markers at 2x speed
  python test.py --model-path checkpoints/best_model.pth --debug --show-boundary -s 2
  
  # Test from top view
  python test.py --model-path checkpoints/best_model.pth --view top
		"""
	)
	
	parser.add_argument('--model-path', type=str, required=True,
					   help='Path to trained model checkpoint')
	parser.add_argument('--episodes', type=int, default=10,
					   help='Number of test episodes (default: 10)')
	parser.add_argument('--max-steps', type=int, default=5000,
					   help='Maximum steps per episode (default: 5000)')
	parser.add_argument('--device', type=str, default='cuda',
					   help='Device (cuda/cpu)')
	parser.add_argument('--view', type=str, choices=['front', 'top', 'side'], default='front',
					   help='Camera view (default: front)')
	parser.add_argument('--randomize', action='store_true',
					   help='Randomize initial rod positions and orientations')
	parser.add_argument('--hard', action='store_true',
					   help='Enable hard mode (rods may be flat)')
	parser.add_argument('--debug', action='store_true',
					   help='Show debug information during testing')
	parser.add_argument('--show-boundary', action='store_true',
					   help='Show boundary markers')
	parser.add_argument('--speed', type=float, default=1.0,
					   help='Simulation speed multiplier (default: 1.0)')
	parser.add_argument('--temperature', type=float, default=1.0,
					   help='Temperature for softmax (default: 1.0). Higher values encourage exploration.')
	parser.add_argument('--no-op-threshold', type=int, default=50,
					   help='Number of consecutive no-ops before forcing exploration (default: 50)')
	parser.add_argument('--easy', action='store_true',
					   help='Easy mode: only use xyz movement + gripper control')
	parser.add_argument('--save-dir', type=str, default='test_env_results',
					   help='Directory to save test results')
	parser.add_argument('--all-modes', action='store_true',
					   help='Test all environment modes (default, randomized, hard)')
	parser.add_argument('--step-by-step', action='store_true',
					   help='Run step-by-step, waiting for user input after each step')
	parser.add_argument('--no-render', action='store_true',
					   help='Disable rendering (run in headless mode)')
	parser.add_argument('--seed', type=int, default=None,
					   help='Random seed (default: random)')
	
	args = parser.parse_args()
	
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	
	os.makedirs(args.save_dir, exist_ok=True)
	
	# Test configurations
	if args.all_modes:
		test_configs = [
			{'name': 'default', 'randomize': False, 'hard': False},
			{'name': 'randomized', 'randomize': True, 'hard': False},
			{'name': 'hard', 'randomize': False, 'hard': True},
			{'name': 'randomized_hard', 'randomize': True, 'hard': True}
		]
	else:
		test_configs = [
			{'name': 'test', 'randomize': args.randomize, 'hard': args.hard}
		]
	
	all_results = {}
	
	for config in test_configs:
		print(f"\n{'#'*60}")
		print(f"Testing mode: {config['name']}")
		print(f"  Randomize: {config['randomize']}")
		print(f"  Hard: {config['hard']}")
		print(f"{'#'*60}")
		
		env_config = {
			'randomize': config['randomize'],
			'hard': config['hard'],
			'easy': args.easy
		}
		
		results = test_model(
			env_config=env_config,
			model_path=args.model_path,
			device=device,
			num_episodes=args.episodes,
			max_steps=args.max_steps,
			view=args.view,
			debug=args.debug,
			show_boundary=args.show_boundary,
			speed=args.speed,
			temperature=args.temperature,
			no_op_threshold=args.no_op_threshold,
			easy=args.easy
		)
		
		if results is not None:
			all_results[config['name']] = results
			
			# Save results for this mode (pickle format)
			mode_save_path = os.path.join(args.save_dir, f"{config['name']}_results.pkl")
			with open(mode_save_path, 'wb') as f:
				pickle.dump(results, f)
			print(f"\nResults for {config['name']} mode saved to: {mode_save_path}")
			
			# Save detailed text log
			mode_log_path = os.path.join(args.save_dir, f"{config['name']}_results.log")
			with open(mode_log_path, 'w') as f:
				f.write("="*60 + "\n")
				f.write(f"Test Results - Mode: {config['name']}\n")
				f.write("="*60 + "\n\n")
				
				f.write("Model Information:\n")
				f.write("-"*40 + "\n")
				model_info = results['model_info']
				f.write(f"  Epoch: {model_info['epoch']}\n")
				f.write(f"  Train Loss: {model_info['train_loss']:.6f}\n")
				f.write(f"  Val Loss: {model_info['val_loss']:.6f}\n")
				f.write(f"  Total Parameters: {model_info['total_params']:,}\n")
				f.write(f"  Easy Mode: {args.easy}\n")
				f.write(f"  Temperature: {args.temperature}\n")
				f.write(f"  No-op Threshold: {args.no_op_threshold}\n\n")
				
				f.write("Overall Statistics:\n")
				f.write("-"*40 + "\n")
				f.write(f"  Success Rate: {results['success_rate']:.2%}\n")
				f.write(f"  Average Steps: {results['avg_steps']:.1f}\n")
				f.write(f"  Connection Rate: {results['connect_rate']:.2%}\n")
				f.write(f"  Hit Target Rate: {results['hit_rate']:.2%}\n")
				f.write(f"  Boundary Violation Rate: {results['violation_rate']:.2%}\n\n")
				
				f.write("Environment Configuration:\n")
				f.write("-"*40 + "\n")
				f.write(f"  Randomize: {config['randomize']}\n")
				f.write(f"  Hard: {config['hard']}\n")
				f.write(f"  Easy: {args.easy}\n")
				f.write(f"  Max Steps: {args.max_steps}\n")
				f.write(f"  Seed: {args.seed if args.seed is not None else 'Random'}\n\n")
				
				f.write("Episode Results:\n")
				f.write("-"*40 + "\n")
				for i, ep_result in enumerate(results['episode_results']):
					f.write(f"\nEpisode {i+1}:\n")
					f.write(f"  Success: {'✓' if ep_result['success'] else '✗'}\n")
					f.write(f"  Steps: {ep_result['steps']}\n")
					f.write(f"  Connected: {'✓' if ep_result['connected'] else '✗'}\n")
					f.write(f"  Hit Target: {'✓' if ep_result['hit_target'] else '✗'}\n")
					f.write(f"  Boundary Violation: {'✓' if ep_result['boundary_violation'] else '✗'}\n")
					if 'error' in ep_result:
						f.write(f"  Error: {ep_result['error']}\n")
				
				f.write("\n" + "="*60 + "\n")
				f.write(f"Generated by test.py\n")
				f.write("="*60 + "\n")
			
			print(f"Text log saved to: {mode_log_path}")
	
	# Plot success rates if multiple modes tested
	if len(all_results) > 1:
		plot_path = os.path.join(args.save_dir, 'success_rates.png')
		plot_success_rates(all_results, plot_path)
		
		# Save all results together
		all_results_path = os.path.join(args.save_dir, 'all_results.pkl')
		with open(all_results_path, 'wb') as f:
			pickle.dump(all_results, f)
		print(f"\nAll results saved to: {all_results_path}")
	
	print("\n" + "="*60)
	print("Testing completed!")
	print("="*60)


if __name__ == "__main__":
	main()
