import numpy as np
import pickle
import os
import time
import pybullet as p
from envs import ArmEnv


class DataCollector:
	BASE_DIR = 'data'
	MIN_STEPS = 10
	MAX_STEPS = 5000
	POS_SPEED = 0.5
	ROT_SPEED = 0.5
	
	def __init__(self, dataset_type='train', debug=False, randomize=False, show_boundary=False, hard=False):
		self.env = ArmEnv(render=True, verbose=False, debug=debug, 
						 randomize=randomize, show_bnd=show_boundary, hard=hard)
		
		if dataset_type not in ['train', 'val']:
			raise ValueError(f"dataset_type must be 'train' or 'val', got '{dataset_type}'")
		self.dataset_type = dataset_type
		
		self.save_dir = os.path.join(self.BASE_DIR, dataset_type)
		self.pos_speed = self.POS_SPEED
		self.rot_speed = self.ROT_SPEED
		
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		
		self.existing_count = self._count_existing_trajectories()
	
	def _count_existing_trajectories(self):
		if not os.path.exists(self.save_dir):
			return 0
		
		files = [f for f in os.listdir(self.save_dir) 
			 if f.startswith('trajectory_') and f.endswith('.pkl')]
		return len(files)
	
	def _ask_dataset_type(self):
		print("\n" + "=" * 60)
		print("Select dataset type:")
		print("  1. Training data (will be saved to data/train/)")
		print("  2. Validation data (will be saved to data/val/)")
		print("=" * 60)
		
		while True:
			choice = input("\nEnter your choice (1 or 2): ").strip()
			if choice == '1':
				return 'train'
			elif choice == '2':
				return 'val'
			else:
				print("Invalid choice. Please enter 1 or 2.")
	
	def _get_action_from_keyboard(self):
		keys = p.getKeyboardEvents()
		action = np.zeros(7)
		
		for key, event in keys.items():
			if not (event & p.KEY_IS_DOWN):
				continue
			
			if key in [ord('c'), ord('C')]:
				action[2] += self.pos_speed
			elif key in [ord('z'), ord('Z')]:
				action[2] -= self.pos_speed
			elif key == p.B3G_LEFT_ARROW:
				action[0] -= self.pos_speed
			elif key == p.B3G_RIGHT_ARROW:
				action[0] += self.pos_speed
			elif key == p.B3G_UP_ARROW:
				action[1] += self.pos_speed
			elif key == p.B3G_DOWN_ARROW:
				action[1] -= self.pos_speed
			elif key in [ord('j'), ord('J')]:
				action[3] += self.rot_speed
			elif key in [ord('l'), ord('L')]:
				action[3] -= self.rot_speed
			elif key in [ord('k'), ord('K')]:
				action[4] += self.rot_speed
			elif key in [ord('i'), ord('I')]:
				action[4] -= self.rot_speed
			elif key in [ord('u'), ord('U')]:
				action[5] += self.rot_speed
			elif key in [ord('o'), ord('O')]:
				action[5] -= self.rot_speed
			elif key == p.B3G_SPACE:
				action[6] = -1.0
			elif key in [ord('b'), ord('B')]:
				action[6] = 1.0
		
		return action
	
	def _handle_view_switch(self, keys, current_view):
		view_keys = {'front': ord('1'), 'top': ord('2'), 'side': ord('3')}
		for view, vk in view_keys.items():
			if vk in keys and keys[vk] & p.KEY_WAS_TRIGGERED and view != current_view:
				self.env.set_camera_view(view)
				return view
		return current_view
	
	def _print_progress(self, step, info):
		if step % 100 == 0:
			st = f"conn:{info['conn']} hit:{info['hit']}"
			print(f"\rsteps: {step:4d}, [{st}]", end="")
	
	def _print_result(self, success, step, info):
		print(f"\n{'='*60}")
		if success:
			print(f"Trajectory completed successfully!")
		else:
			print(f"Trajectory failed!")
			reason = 'Boundary violation' if info['bnd_vio'] else 'Unknown'
			print(f"  Reason: {reason}")
		print(f"  Steps: {step}")
		print(f"  Connected: {info['conn']}")
		print(f"  Hit target: {info['hit']}")
	
	def _print_controls(self):
		print("\n" + "=" * 60)
		print("=== Starting trajectory collection ===")
		print("=" * 60)
		print("Keyboard controls:")
		print("  Arm movement (IK control):")
		print("    C/Z: move end-effector up/down")
		print("    Arrow keys: move end-effector left/right/forward/backward")
		print("  Gripper rotation:")
		print("    J/L: rotate around X axis (roll)")
		print("    I/K: rotate around Y axis (pitch)")
		print("    U/O: rotate around Z axis (yaw)")
		print("  Gripper control:")
		print("    Space: close gripper")
		print("    B: open gripper")
		print("  View switching:")
		print("    1/2/3: front/top/side view")
		print("  Other:")
		print("    ESC: exit (no save)")
		print("    Auto-save on success, no save on failure")
		print("=" * 60)
	
	def _collect_loop(self, traj_id):
		traj = {'observations': [], 'actions': []}
		step = 0
		view = 'front'
		
		while step < self.MAX_STEPS:
			obs = self.env.get_obs()
			action = self._get_action_from_keyboard()
			done, info = self.env.step(action)
			
			traj['observations'].append(obs)
			traj['actions'].append(action)
			
			self._print_progress(step, info)
			time.sleep(1./60.)
			step += 1
			
			if done:
				self._print_result(info['hit'], step, info)
				return traj if info['hit'] else None
			
			keys = p.getKeyboardEvents()
			view = self._handle_view_switch(keys, view)
			
			if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
				print("\n\nUser exited (not saved)")
				return None
		
		print(f"\n\nReached max step limit ({self.MAX_STEPS})")
		return None
	
	def collect_single_trajectory(self):
		print("\n" + "=" * 60)
		print("PRML Project - Arm Imitation Learning Data Collection")
		print("=" * 60)
		print("\nTask:")
		print("  1. Connect two rods")
		print("  2. Grasp the combined rod")
		print("  3. Strike the target on the wall")
		print(f"\nDataset type: {self.dataset_type.upper()}")
		print(f"Save directory: {os.path.abspath(self.save_dir)}")
		print(f"Existing trajectories: {self.existing_count}")
		print(f"\nNotes:")
		print(f"  - Trajectories are automatically saved as separate files")
		print(f"  - Press ESC to exit (no save)")
		print(f"  - Auto-save on successful target hit")
		print(f"  - No save on failure")
		print(f"  - Max steps per trajectory: {self.MAX_STEPS}")
		print("=" * 60)
		
		self._print_controls()
		tid = self.existing_count + 1
		traj = self._collect_loop(tid)
		
		if traj is None:
			print("\n\nData collection ended (not saved)")
			return
		
		if len(traj['observations']) < self.MIN_STEPS:
			print(f"\n\n⚠ Trajectory too short (<{self.MIN_STEPS} steps), discarded")
			return
		
		self._save_traj(traj, tid)
		print(f"\n" + "=" * 60)
		print("Data collection completed successfully")
		print(f"All data saved in: {os.path.abspath(self.save_dir)}")
		print("=" * 60)
	
	def _save_traj(self, traj, traj_id):
		fn = os.path.join(self.save_dir, f'trajectory_{traj_id:03d}.pkl')
		
		scene_info = {
			'wall_pos': list(self.env.wall_pos) if hasattr(self.env, 'wall_pos') else [0.0, 1.0, 0.5],
			'wall_orn': list(self.env.wall_orn) if hasattr(self.env, 'wall_orn') else [0, 0, 0, 1]
		}
		
		traj['traj_id'] = traj_id
		traj['length'] = len(traj['observations'])
		traj['scene_info'] = scene_info
		
		with open(fn, 'wb') as f:
			pickle.dump(traj, f)
		
		print(f"✓ Trajectory saved: {fn}")
		print(f"  Length: {traj['length']} steps")
	
	def close(self):
		self.env.close()


def main():
	import argparse
	
	parser = argparse.ArgumentParser(description='PRML Project - Arm Data Collection')
	parser.add_argument('-t', '--type', type=str, choices=['train', 'val'], default=None,
					   help='Dataset type (train or val). If not specified, will ask interactively')
	parser.add_argument('-d', '--debug', action='store_true', help='Show debug info')
	parser.add_argument('-r', '--randomize', action='store_true', help='Randomize initial position')
	parser.add_argument('-b', '--show-boundary', action='store_true', help='Show boundary markers')
	parser.add_argument('--hard', action='store_true', help='Enable hard mode (rods may be flat)')
	
	args = parser.parse_args()
	
	if args.type is None:
		dataset_type = DataCollector._ask_dataset_type(None)
	else:
		dataset_type = args.type
	
	collector = DataCollector(dataset_type=dataset_type, debug=args.debug, 
							 randomize=args.randomize, show_boundary=args.show_boundary, hard=args.hard)
	
	try:
		collector.collect_single_trajectory()
	except KeyboardInterrupt:
		print("\n\nProgram interrupted")
	except Exception as e:
		print(f"\nError: {e}")
		import traceback
		traceback.print_exc()
	finally:
		collector.close()


if __name__ == "__main__":
	main()
