import numpy as np
import pickle
import os
import argparse
import time
from typing import Dict, List
from envs import ArmEnv


class TrajectoryReplayer:
	BASE_DIR = 'data'
	
	OBS_JOINT_POS = 9
	OBS_JOINT_VEL = 9
	OBS_END_POS = 3
	OBS_END_ORN = 4
	OBS_ROD_STATE = 13
	OBS_TOTAL = 38
	
	WARN_END_POS_THRESHOLD = 0.05
	WARN_END_ORN_THRESHOLD = 0.05
	WARN_JOINT_THRESHOLD = 0.1
	WARN_ROD_THRESHOLD = 0.5
	
	def __init__(self, debug=False, show_boundary=False, speed=1.0, render=True):
		self.env = ArmEnv(render=render, verbose=False, debug=debug, 
						 randomize=False, show_bnd=show_boundary)
		self.speed = speed
		self.debug = debug
		self.render = render
		print(f"\nTrajectory Replayer initialized")
		print(f"  Debug mode: {debug}")
		print(f"  Show boundary: {show_boundary}")
		print(f"  Playback speed: {speed}x")
		print(f"  Render mode: {render}")
		
		np.random.seed(42)
	
	def load_trajectory(self, filepath) -> Dict:
		if not os.path.exists(filepath):
			raise FileNotFoundError(f"Trajectory file not found: {filepath}")
		
		print(f"Loading trajectory from: {filepath}")
		with open(filepath, 'rb') as f:
			traj = pickle.load(f)
		
		print(f"  Trajectory ID: {traj.get('traj_id', 'N/A')}")
		print(f"  Length: {traj['length']} steps")
		print(f"  Number of observations: {len(traj['observations'])}")
		print(f"  Number of actions: {len(traj['actions'])}")
		
		return traj
	
	def replay(self, traj: Dict, view: str = 'front', compare_with_original: bool = True) -> List[np.ndarray]:
		print("\n" + "=" * 60)
		print("Starting trajectory replay")
		print("=" * 60)
		print(f"  View: {view}")
		print(f"  Speed: {self.speed}x")
		print(f"  Compare with original: {compare_with_original}")
		print("  Press ESC to exit early")
		print("=" * 60)
		
		self.env.set_camera_view(view)
		
		observations = traj['observations']
		actions = traj['actions']
		length = min(len(observations), len(actions))
		
		self._load_initial_state(observations[0])
		
		print(f"\nReplaying {length} steps...")
		
		replay_observations = []
		
		if compare_with_original:
			print("\nReal-time Error Statistics (updated every 100 steps):")
			print("-" * 60)
		
		step = 0
		info = None
		try:
			for step in range(length):
				action = np.array(actions[step])
				done, info = self.env.step(action)
				
				current_obs = self.env.get_obs()
				replay_observations.append(current_obs)
				
				if (step + 1) % 100 == 0 and compare_with_original:
					self._print_realtime_comparison(
						observations[:len(replay_observations)], 
						replay_observations, 
						step + 1,
						info
					)
				
				if self.render:
					time.sleep(1./ (self.speed * 60))
				
				if done:
					print(f"\n\nTrajectory completed early at step {step + 1}")
					print(f"  Reached height: {info['reached_height']}")
					print(f"  Boundary violation: {info['bnd_vio']}")
					break
			
			if step + 1 == length:
				print(f"\n\nReplay completed successfully!")
				print(f"  Total steps: {length}")
				if info:
					print(f"  Final reached height: {info['reached_height']}")
			
			if compare_with_original and len(replay_observations) > 0:
				self._print_comparison(observations[:len(replay_observations)], replay_observations)
		
		except KeyboardInterrupt:
			print("\n\nReplay interrupted by user")
		
		except Exception as e:
			print(f"\n\nError during replay: {e}")
			if self.debug:
				import traceback
				traceback.print_exc()
		
		return replay_observations
	
	def _parse_observation_structure(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
		return {
			'joint_pos': obs[0:self.OBS_JOINT_POS],
			'joint_vel': obs[self.OBS_JOINT_POS:self.OBS_JOINT_POS+self.OBS_JOINT_VEL],
			'end_pos': obs[self.OBS_JOINT_POS+self.OBS_JOINT_VEL:self.OBS_JOINT_POS+self.OBS_JOINT_VEL+self.OBS_END_POS],
			'end_orn': obs[self.OBS_JOINT_POS+self.OBS_JOINT_VEL+self.OBS_END_POS:self.OBS_JOINT_POS+self.OBS_JOINT_VEL+self.OBS_END_POS+self.OBS_END_ORN],
			'rod_state': obs[self.OBS_JOINT_POS+self.OBS_JOINT_VEL+self.OBS_END_POS+self.OBS_END_ORN:]
		}
	
	def _print_realtime_comparison(self, original_obs: np.ndarray, replay_obs: List[np.ndarray], 
								current_step: int, info: Dict) -> None:
		original_obs = np.array(original_obs)
		replay_obs = np.array(replay_obs)
		
		diff = np.abs(original_obs - replay_obs)
		
		end_pos_error = np.linalg.norm(diff[-1, self.OBS_JOINT_POS+self.OBS_JOINT_VEL:self.OBS_JOINT_POS+self.OBS_JOINT_VEL+self.OBS_END_POS])
		end_orn_error = np.linalg.norm(diff[-1, self.OBS_JOINT_POS+self.OBS_JOINT_VEL+self.OBS_END_POS:self.OBS_JOINT_POS+self.OBS_JOINT_VEL+self.OBS_END_POS+self.OBS_END_ORN])
		joint_pos_error = np.mean(diff[-1, 0:self.OBS_JOINT_POS])
		
		end_pos_cum_error = np.sum(np.linalg.norm(
			diff[:, self.OBS_JOINT_POS+self.OBS_JOINT_VEL:self.OBS_JOINT_POS+self.OBS_JOINT_VEL+self.OBS_END_POS], 
			axis=1
		))
		
		rod_idx = self.OBS_JOINT_POS + self.OBS_JOINT_VEL + self.OBS_END_POS + self.OBS_END_ORN
		rod_error = np.linalg.norm(diff[-1, rod_idx:rod_idx+3])
		
		warn_pos = "!" if end_pos_error > self.WARN_END_POS_THRESHOLD else ""
		warn_orn = "!" if end_orn_error > self.WARN_END_ORN_THRESHOLD else ""
		warn_joint = "!" if joint_pos_error > self.WARN_JOINT_THRESHOLD else ""
		warn_rod = "!" if rod_error > self.WARN_ROD_THRESHOLD else ""
		
		print(f"\rStep {current_step:4d} | "
			  f"End-Pos: {end_pos_error:6.3f}m{warn_pos} | "
			  f"End-Orn: {end_orn_error:6.3f}{warn_orn} | "
			  f"Joint: {joint_pos_error:6.3f}{warn_joint} | "
			  f"Rod: {rod_error:6.3f}m{warn_rod} | "
			  f"Cum: {end_pos_cum_error:8.2f}m | "
			  f"Reach:{info['reached_height']} Bnd:{info['bnd_vio']}", 
			  end="", flush=True)
	
	def _load_initial_state(self, obs):
		import pybullet as p
		
		idx = 0
		
		joint_pos = obs[idx:idx+9]
		idx += 9
		
		joint_vel = obs[idx:idx+9]
		idx += 9
		
		end_pos = obs[idx:idx+3]
		idx += 3
		end_orn = obs[idx:idx+4]
		idx += 4
		
		rod_state = obs[idx:idx+13]
		
		NUM_ARM_JOINTS = 7
		GRIPPER_JOINTS = [9, 10]
		
		gripper_pos = joint_pos[NUM_ARM_JOINTS]
		self.env.gripper_closed = (gripper_pos < self.env.GRIPPER_CLOSED + 0.01)
		
		for i in range(NUM_ARM_JOINTS):
			p.resetJointState(self.env.arm, i, joint_pos[i], joint_vel[i])
		
		for gj_idx, gj in enumerate(GRIPPER_JOINTS):
			p.resetJointState(self.env.arm, gj, joint_pos[NUM_ARM_JOINTS + gj_idx], 0.0)
		
		if self.env.rod:
			p.resetBasePositionAndOrientation(self.env.rod, rod_state[0:3], rod_state[3:7])
			p.resetBaseVelocity(self.env.rod, rod_state[7:10], rod_state[10:13])
		
		for i in range(NUM_ARM_JOINTS):
			p.setJointMotorControl2(self.env.arm, i, p.POSITION_CONTROL, 
								   targetPosition=joint_pos[i], 
								   force=self.env.JOINT_FORCE,
								   maxVelocity=self.env.JOINT_FORCE)
		
		for gj_idx, gj in enumerate(GRIPPER_JOINTS):
			p.setJointMotorControl2(self.env.arm, gj, p.POSITION_CONTROL,
								   targetPosition=joint_pos[NUM_ARM_JOINTS + gj_idx], 
								   force=self.env.GRIPPER_FORCE,
								   maxVelocity=self.env.GRIPPER_FORCE)
		
		for _ in range(10):
			p.stepSimulation()
	
	def _print_comparison(self, original_obs: List[np.ndarray], replay_obs: List[np.ndarray]) -> None:
		print("\n" + "=" * 60)
		print("Trajectory Comparison (Replay vs Original)")
		print("=" * 60)
		
		original_obs = np.array(original_obs)
		replay_obs = np.array(replay_obs)
		
		diff = np.abs(original_obs - replay_obs)
		
		def print_stats(name, data: np.ndarray) -> None:
			if data.ndim == 1:
				data = data.reshape(-1, 1)
			
			mean = np.mean(data, axis=0)
			max_val = np.max(data, axis=0)
			rms = np.sqrt(np.mean(data**2, axis=0))
			
			print(f"   Mean: {mean}")
			print(f"   Max:  {max_val}")
			print(f"   RMS:  {rms}")
		
		print("\n1. End-effector Position Error:")
		print_stats("Position", diff[:, self.OBS_JOINT_POS+self.OBS_JOINT_VEL:self.OBS_JOINT_POS+self.OBS_JOINT_VEL+self.OBS_END_POS])
		
		print("\n2. End-effector Orientation Error (quaternion):")
		print_stats("Orientation", diff[:, self.OBS_JOINT_POS+self.OBS_JOINT_VEL+self.OBS_END_POS:self.OBS_JOINT_POS+self.OBS_JOINT_VEL+self.OBS_END_POS+self.OBS_END_ORN])
		
		print("\n3. Joint Position Error:")
		for i in range(self.OBS_JOINT_POS):
			joint_name = f"Arm {i}" if i < 7 else f"Gripper {i-6}"
			mean = np.mean(diff[:, i])
			max_val = np.max(diff[:, i])
			warn = "!" if mean > self.WARN_JOINT_THRESHOLD else ""
			print(f"   {joint_name}: mean={mean:.6f}, max={max_val:.6f}{warn}")
		
		print("\n4. Rod Position Error:")
		rod_idx = self.OBS_JOINT_POS + self.OBS_JOINT_VEL + self.OBS_END_POS + self.OBS_END_ORN
		print_stats("Rod", diff[:, rod_idx:rod_idx+3])
		
		print("\n5. Overall Statistics:")
		print(f"   Total steps: {len(original_obs)}")
		
		print("\n6. Cumulative Errors:")
		end_pos_cum_error = np.cumsum(np.linalg.norm(diff[:, self.OBS_JOINT_POS+self.OBS_JOINT_VEL:self.OBS_JOINT_POS+self.OBS_JOINT_VEL+self.OBS_END_POS], axis=1))
		percentiles = [0.25, 0.5, 0.75, 1.0]
		for p in percentiles:
			idx_p = min(int(len(end_pos_cum_error) * p), len(end_pos_cum_error) - 1)
			print(f"   At {int(p*100):3d}%: {end_pos_cum_error[idx_p]:.4f}")
		
		print("=" * 60)
	
	def list_trajectories(self, dataset_type: str) -> List[str]:
		save_dir = os.path.join(self.BASE_DIR, dataset_type)
		if not os.path.exists(save_dir):
			return []
		
		files = sorted([f for f in os.listdir(save_dir) 
					   if f.startswith('trajectory_') and f.endswith('.pkl')])
		
		return files
	
	def close(self) -> None:
		self.env.close()


def main():
	parser = argparse.ArgumentParser(
		description='PRML Project - Trajectory Replay',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Replay a specific trajectory
  python replay.py -f data/train/trajectory_001.pkl
  
  # Replay all training trajectories
  python replay.py -t train -a
  
  # Replay with debug info and boundary markers at 0.5x speed
  python replay.py -f data/train/trajectory_001.pkl -d -b -s 0.5
  
  # Replay all validation trajectories at 2x speed
  python replay.py -t val -a -s 2
		"""
	)
	
	parser.add_argument('-b', '--show-boundary', action='store_true', 
					   help='Show boundary markers')
	parser.add_argument('-d', '--debug', action='store_true', 
					   help='Show debug information')
	parser.add_argument('-s', '--speed', type=float, default=1.0,
					   help='Playback speed multiplier (default: 1.0)')
	parser.add_argument('-a', '--all', action='store_true', 
					   help='Replay all trajectories in the specified dataset type')
	parser.add_argument('-t', '--type', type=str, choices=['train', 'val'], default='train',
					   help='Dataset type (train or val). Used with -a to select which dataset to replay')
	parser.add_argument('-f', '--file', type=str, default=None,
					   help='Specific trajectory file to replay (e.g., data/train/trajectory_001.pkl)')
	parser.add_argument('-v', '--view', type=str, choices=['front', 'top', 'side'], default='front',
					   help='Initial camera view (default: front)')
	parser.add_argument('--render', action='store_true', default=True,
					   help='Enable rendering (default: True)')
	parser.add_argument('--no-render', dest='render', action='store_false',
					   help='Disable rendering for headless mode')
	
	args = parser.parse_args()
	
	if args.all and args.file:
		print("Error: Cannot use both -a/--all and -f/--file at the same time")
		return
	
	if not args.all and not args.file:
		print("Error: Must specify either -a/--all to replay all trajectories or -f/--file for a specific file")
		parser.print_help()
		return
	
	replayer = None
	
	try:
		if args.all:
			print(f"\n" + "=" * 60)
			print(f"Replaying ALL trajectories from {args.type.upper()} dataset")
			print("=" * 60)
			
			replayer = TrajectoryReplayer(debug=args.debug, 
										  show_boundary=args.show_boundary, 
										  speed=args.speed,
										  render=args.render)
			
			files = replayer.list_trajectories(args.type)
			
			if not files:
				print(f"\nNo trajectories found in {args.type} dataset")
				return
			
			print(f"\nFound {len(files)} trajectories:")
			for i, f in enumerate(files, 1):
				print(f"  {i}. {f}")
			
			print(f"\nReplaying {len(files)} trajectories...")
			
			for idx, filename in enumerate(files, 1):
				filepath = os.path.join(replayer.BASE_DIR, args.type, filename)
				print(f"\n" + "=" * 60)
				print(f"[{idx}/{len(files)}] Replaying: {filename}")
				print("=" * 60)
				
				try:
					traj = replayer.load_trajectory(filepath)
					replayer.replay(traj, view=args.view)
					
					input("\nPress Enter to continue to next trajectory (or Ctrl+C to exit)...")
				
				except KeyboardInterrupt:
					print(f"\nReplay interrupted by user during trajectory {idx}/{len(files)}")
					break
				
				except Exception as e:
					print(f"\nError replaying {filename}: {e}")
					continue
				
				replayer.env.reset()
			
			print(f"\n" + "=" * 60)
			print(f"Finished replaying trajectories")
			print("=" * 60)
		
		else:
			replayer = TrajectoryReplayer(debug=args.debug, 
										  show_boundary=args.show_boundary, 
										  speed=args.speed,
										  render=args.render)
			
			traj = replayer.load_trajectory(args.file)
			replayer.replay(traj, view=args.view)
	
	except KeyboardInterrupt:
		print("\n\nProgram interrupted by user")
	
	except Exception as e:
		print(f"\nError: {e}")
		import traceback
		traceback.print_exc()
	
	finally:
		if replayer:
			replayer.close()
			print("\nReplayer closed")


if __name__ == "__main__":
	main()
