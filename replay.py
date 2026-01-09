import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm

from envs import ArmEnv


class TrajectoryReplayer:
	SAVE_DIR = 'data'
	
	def __init__(self, render=True, verbose=False, debug=False, show_boundary=False):
		self.env = ArmEnv(render=render, verbose=verbose, debug=debug, show_bnd=show_boundary)
		self.save_dir = self.SAVE_DIR
	
	def list_trajectories(self):
		if not os.path.exists(self.save_dir):
			print(f"Data directory '{self.save_dir}' not found")
			return []
		
		files = sorted([f for f in os.listdir(self.save_dir) 
					   if f.startswith('trajectory_') and f.endswith('.pkl')])
		
		if not files:
			print(f"No trajectory files found in '{self.save_dir}'")
			return []
		
		print(f"\nFound {len(files)} trajectory file(s) in '{self.save_dir}':\n")
		
		traj_list = []
		for fn in files:
			path = os.path.join(self.save_dir, fn)
			try:
				with open(path, 'rb') as f:
					traj = pickle.load(f)
				
				md = traj.get('metadata', {})
				print(f"  {fn}:")
				print(f"    Length: {md.get('length', 'N/A')} steps")
				print(f"    Total reward: {md.get('total_reward', 'N/A'):.2f}" 
					  if isinstance(md.get('total_reward'), (int, float)) else f"    Total reward: {md.get('total_reward', 'N/A')}")
				print(f"    Connected: {md.get('final_connected', 'N/A')}")
				print(f"    Hit target: {md.get('final_hit', 'N/A')}")
				print()
				
				traj_list.append(fn)
			except Exception as e:
				print(f"  {fn}: Error reading file - {e}\n")
		
		return traj_list
	
	def load_trajectory(self, filename):
		path = os.path.join(self.save_dir, filename) if not os.path.isabs(filename) else filename
		
		if not os.path.exists(path):
			print(f"File not found: {path}")
			return None
		
		try:
			with open(path, 'rb') as f:
				traj = pickle.load(f)
			
			print(f"\nSuccessfully loaded: {filename}")
			md = traj.get('metadata', {})
			print(f"  Length: {md.get('length', 'N/A')} steps")
			print(f"  Total reward: {md.get('total_reward', 'N/A'):.2f}" 
				  if isinstance(md.get('total_reward'), (int, float)) else f"  Total reward: {md.get('total_reward', 'N/A')}")
			print(f"  Connected: {md.get('final_connected', 'N/A')}")
			print(f"  Hit target: {md.get('final_hit', 'N/A')}")
			
			return traj
		except Exception as e:
			print(f"Error loading trajectory: {e}")
			return None
	
	def replay_trajectory(self, traj, slow_down=False, show_info=True):
		if traj is None:
			print("No trajectory to replay")
			return
		
		observations = traj.get('observations', [])
		actions = traj.get('actions', [])
		rewards = traj.get('rewards', [])
		infos = traj.get('info', [])
		
		if not observations or not actions:
			print("Invalid trajectory data")
			return
		
		n_steps = len(observations)
		print(f"\n{'='*60}")
		print(f"Replaying trajectory ({n_steps} steps)...")
		print(f"{'='*60}")
		if show_info:
			print("\nPress Ctrl+C to stop early")
		
		try:
			for step in tqdm(range(n_steps), desc="Replaying", disable=not show_info):
				obs, reward, done, info = self.env.step(actions[step])
				
				if show_info and (step + 1) % 100 == 0:
					print(f"\nStep {step + 1}/{n_steps}:")
					print(f"  Reward: {reward:.2f}")
					print(f"  Connected: {info['conn']}")
					print(f"  Hit: {info['hit']}")
				
				if slow_down:
					import time
					time.sleep(0.01)
				
				if done:
					print(f"\nTrajectory ended early at step {step + 1}")
					break
			
			print(f"\n{'='*60}")
			print("Replay completed!")
			if rewards:
				print(f"Total reward: {sum(rewards):.2f}")
			if infos:
				last_info = infos[-1] if infos else {}
				print(f"Final connected: {last_info.get('conn', False)}")
				print(f"Final hit: {last_info.get('hit', False)}")
			print(f"{'='*60}")
		
		except KeyboardInterrupt:
			print(f"\n\nReplay interrupted by user at step {step + 1}")
	
	def replay_single(self, filename, slow_down=False, show_info=True):
		traj = self.load_trajectory(filename)
		if traj:
			self.replay_trajectory(traj, slow_down=slow_down, show_info=show_info)
	
	def replay_all(self, slow_down=False, show_info=False):
		traj_files = self.list_trajectories()
		
		if not traj_files:
			return
		
		print(f"\n{'='*60}")
		print(f"Replaying {len(traj_files)} trajectory(s)...")
		print(f"{'='*60}")
		
		for i, filename in enumerate(traj_files):
			print(f"\n[{i+1}/{len(traj_files)}] Loading {filename}...")
			traj = self.load_trajectory(filename)
			if traj:
				self.env.reset()
				self.replay_trajectory(traj, slow_down=slow_down, show_info=show_info)
			
			print("\n" + "-" * 60)
	
	def compare_rewards(self, filename):
		traj = self.load_trajectory(filename)
		if traj is None:
			return
		
		actions = traj.get('actions', [])
		original_rewards = traj.get('rewards', [])
		
		if not actions:
			print("No actions in trajectory")
			return
		
		print(f"\n{'='*60}")
		print("Comparing original and replay rewards...")
		print(f"{'='*60}")
		
		self.env.reset()
		replay_rewards = []
		infos = []
		
		for step, action in enumerate(actions):
			_, reward, done, info = self.env.step(action)
			replay_rewards.append(reward)
			infos.append(info)
			
			if done:
				break
		
		print(f"\nOriginal trajectory:")
		print(f"  Steps: {len(original_rewards)}")
		print(f"  Total reward: {sum(original_rewards):.2f}")
		
		print(f"\nReplay:")
		print(f"  Steps: {len(replay_rewards)}")
		print(f"  Total reward: {sum(replay_rewards):.2f}")
		
		print(f"\nDifference:")
		print(f"  Step difference: {len(replay_rewards) - len(original_rewards)}")
		print(f"  Reward difference: {sum(replay_rewards) - sum(original_rewards):.2f}")
		
		print(f"{'='*60}")
	
	def close(self):
		self.env.close()


def main():
	parser = argparse.ArgumentParser(description='Replay .pkl trajectory files')
	parser.add_argument('trajectory', type=str, nargs='?', 
					   help='Trajectory filename (e.g., trajectory_001.pkl) or path')
	parser.add_argument('--list', '-l', action='store_true',
					   help='List all available trajectory files')
	parser.add_argument('--all', '-a', action='store_true',
					   help='Replay all trajectory files')
	parser.add_argument('--compare', '-c', action='store_true',
					   help='Compare original and replay rewards')
	parser.add_argument('--slow', '-s', action='store_true',
					   help='Slow replay (for observation)')
	parser.add_argument('--verbose', '-v', action='store_true',
					   help='Show detailed information')
	parser.add_argument('--debug', '-d', action='store_true',
					   help='Show debug information')
	parser.add_argument('--show-boundary', '-b', action='store_true',
					   help='Show boundary markers')
	parser.add_argument('--no-render', action='store_true',
					   help='Do not render environment (replay without display)')
	
	args = parser.parse_args()
	
	replayer = TrajectoryReplayer(
		render=not args.no_render,
		verbose=args.verbose,
		debug=args.debug,
		show_boundary=args.show_boundary
	)
	
	try:
		if args.list:
			replayer.list_trajectories()
		
		elif args.all:
			replayer.replay_all(slow_down=args.slow, show_info=args.verbose)
		
		elif args.trajectory:
			if args.compare:
				replayer.compare_rewards(args.trajectory)
			else:
				replayer.replay_single(
					args.trajectory,
					slow_down=args.slow,
					show_info=args.verbose
				)
		
		else:
			print("Please specify a trajectory file to replay, or use --list to see available files")
			print("Use --help for help information")
	
	except KeyboardInterrupt:
		print("\n\nInterrupted by user")
	
	except Exception as e:
		print(f"\nError: {e}")
		import traceback
		traceback.print_exc()
	
	finally:
		replayer.close()


if __name__ == "__main__":
	main()