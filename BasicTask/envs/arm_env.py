import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import Tuple, Dict


class ArmEnv:
	ROD_L = 0.2
	ROD_R = 0.02
	ROD_M = 0.25
	TGT_HEIGHT = 0.5
	BND_R = 1.0
	
	NUM_ARM_JOINTS = 7
	NUM_MOVABLE_JOINTS = 9
	END_IDX = 11
	GRIPPER_JOINTS = [9, 10]
	INIT_JOINTS = [0, -0.785, 0, -2.356, 0, 1.571, 0]
	GRIPPER_OPEN = 0.04
	GRIPPER_CLOSED = 0.0
	
	MAX_POS_STEP = 0.02
	JOINT_FORCE = 500
	GRIPPER_FORCE = 300
	
	CAMERA_CONFIGS = {
		'front': {'distance': 1.5, 'yaw': 0, 'pitch': -30, 'target': [0, 0, 0.3]},
		'top': {'distance': 1.5, 'yaw': 0, 'pitch': -89.9, 'target': [0, 0, 0.3]},
		'side': {'distance': 1.5, 'yaw': 90, 'pitch': -30, 'target': [0, 0, 0.3]},
	}
	
	def __init__(self, render=True, verbose=False, debug=False, show_bnd=False, randomize=False, seed=None):
		self.render = render
		self.verbose = verbose
		self.debug = debug
		self.show_bnd = show_bnd
		self.randomize = randomize
		if seed is None:
			seed = int(time.time_ns())
		self.seed = seed
		self.rng = np.random.default_rng(seed)
		self.gripper_helpers = {}
		if self.verbose:
			print(f"Random seed: {seed}")
		
		self.pc = p.connect(p.GUI if render else p.DIRECT)
		p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

		if render:
			p.resetDebugVisualizerCamera(1.5, 0, 0, [0.5, 0, 0.3])
			p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
			p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
		
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.81)
		p.loadURDF("plane.urdf")
		
		self.arm = None
		self.rod = None
		self.tgt_id = None
		self.gripper_closed = False
		
		if self.show_bnd:
			self._create_bnd_marker()
		
		self.reset()
	
	def _create_arm(self):
		self.arm = p.loadURDF("franka_panda/panda.urdf", [0, -0.25, 0], [0, 0, 0, 1], useFixedBase=True)
		
		for link in range(p.getNumJoints(self.arm)):
			p.setCollisionFilterGroupMask(self.arm, link, collisionFilterGroup=1, collisionFilterMask=1)
		p.setCollisionFilterGroupMask(self.arm, -1, collisionFilterGroup=1, collisionFilterMask=1)
		
		for gj in self.GRIPPER_JOINTS:
			p.changeDynamics(self.arm, gj, lateralFriction=1.5, spinningFriction=0.1, rollingFriction=0.01)
		
		if self.render and self.debug:
			self._add_gripper_helpers()
		
		return self.arm
	
	def _add_gripper_helpers(self):
		end_pos, end_orn = p.getLinkState(self.arm, 11)[:2]
		mat = np.array(p.getMatrixFromQuaternion(end_orn)).reshape(3, 3)
		
		for axis, color, length in [('x', [1, 0, 0], 0.3), ('y', [0, 1, 0], 0.3), ('z', [1, 1, 1], 0.5)]:
			world_vec = mat @ {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[axis]
			self.gripper_helpers[f'{axis}_axis'] = p.addUserDebugLine(
				end_pos, end_pos + world_vec * length, color, lineWidth=3 + (axis == 'z'), lifeTime=0
			)
	
	def _update_gripper_helpers(self):
		if not self.gripper_helpers:
			return
		
		end_pos, end_orn = p.getLinkState(self.arm, 11)[:2]
		mat = np.array(p.getMatrixFromQuaternion(end_orn)).reshape(3, 3)
		
		for axis, color, length in [('x', [1, 0, 0], 0.3), ('y', [0, 1, 0], 0.3), ('z', [1, 1, 1], 0.5)]:
			world_vec = mat @ {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[axis]
			p.addUserDebugLine(end_pos, end_pos + world_vec * length, color, 
							  lineWidth=3 + (axis == 'z'), lifeTime=0,
							  replaceItemUniqueId=self.gripper_helpers[f'{axis}_axis'])
	
	def _create_rod(self, color, pos, length=ROD_L, mass=ROD_M, orn=[0, 0, 0, 1]):
		v = p.createVisualShape(p.GEOM_CYLINDER, radius=self.ROD_R, length=length, rgbaColor=color)
		c = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.ROD_R, height=length)
		rod_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=c,
								   baseVisualShapeIndex=v, basePosition=pos, baseOrientation=orn)
		p.changeDynamics(rod_id, -1, lateralFriction=0.9, spinningFriction=0.1, rollingFriction=0.01,
						linearDamping=0.04, angularDamping=0.04, restitution=0.1)
		return rod_id
	
	def _create_bnd_marker(self):
		n, r = 36, self.BND_R
		seg_l = 2 * np.pi * r / n
		he = [seg_l / 2, 0.01, 0.01]
		
		for i in range(n):
			ang = 2 * np.pi * i / n
			x, y, z = r * np.cos(ang), r * np.sin(ang), 0.01
			bv = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[1, 0, 0, 0.7])
			q = [0, 0, np.sin((ang + np.pi/2) / 2), np.cos((ang + np.pi/2) / 2)]
			p.createMultiBody(baseMass=0, baseVisualShapeIndex=bv, basePosition=[x, y, z], baseOrientation=q)
	
	def _check_bnd_vio(self) -> bool:
		pos = p.getBasePositionAndOrientation(self.rod)[0]
		return np.linalg.norm(np.array(pos[:2])) > self.BND_R
	
	def _check_height_reached(self) -> bool:
		pos, _ = p.getBasePositionAndOrientation(self.rod)
		return pos[2] > self.TGT_HEIGHT
	
	def _is_done(self) -> bool:
		return self._check_height_reached() or self._check_bnd_vio()
	
	def reset(self):
		if self.arm is None:
			self._create_arm()
		
		if self.rod is not None:
			p.resetBasePositionAndOrientation(self.rod, [0, 0, 10], [0, 0, 0, 1])
			p.removeBody(self.rod)
		
		self.gripper_closed = False
		
		for i in range(self.NUM_ARM_JOINTS):
			p.resetJointState(self.arm, i, self.INIT_JOINTS[i], 0)
			p.setJointMotorControl2(self.arm, i, p.POSITION_CONTROL, targetPosition=self.INIT_JOINTS[i], force=self.JOINT_FORCE)
		
		for gj in self.GRIPPER_JOINTS:
			p.resetJointState(self.arm, gj, self.GRIPPER_OPEN, 0)
			p.setJointMotorControl2(self.arm, gj, p.POSITION_CONTROL, targetPosition=self.GRIPPER_OPEN, force=self.GRIPPER_FORCE)
		
		orn = [0, 0, 0, 1]
		
		if self.randomize:
			pos = [self.rng.uniform(-0.4, 0.4), self.rng.uniform(0.2, 0.4), self.ROD_L / 2]
		else:
			pos = [0.0, 0.3, self.ROD_L / 2]
		
		self.rod = self._create_rod([1, 0, 0, 1], pos, orn=orn)
		
		p.removeAllUserDebugItems()
		self.gripper_helpers = {}
		
		if self.render and self.debug:
			self._add_gripper_helpers()
		
		return self.get_obs()
	
	def get_obs(self) -> np.ndarray:
		joint_indices = list(range(self.NUM_ARM_JOINTS)) + self.GRIPPER_JOINTS
		joint_states = p.getJointStates(self.arm, joint_indices)
		joint_pos = [s[0] for s in joint_states]
		joint_vel = [s[1] for s in joint_states]
		
		end_pos, end_orn = p.getLinkState(self.arm, self.END_IDX)[:2]
		
		rod_pos, rod_orn = p.getBasePositionAndOrientation(self.rod)
		rod_vel, rod_ang_v = p.getBaseVelocity(self.rod)
		
		if self.verbose:
			print(f"Debug get_obs:")
			print(f"  joint_pos length: {len(joint_pos)}")
			print(f"  joint_vel length: {len(joint_vel)}")
			print(f"  end_pos length: {len(list(end_pos))}")
			print(f"  end_orn length: {len(list(end_orn))}")
			print(f"  rod state length: {len(list(rod_pos) + list(rod_orn) + list(rod_vel) + list(rod_ang_v))}")
		
		obs = np.array(joint_pos + joint_vel + list(end_pos) + list(end_orn) +
					  list(rod_pos) + list(rod_orn) + list(rod_vel) + list(rod_ang_v))
		
		return obs
	
	def step(self, action: np.ndarray) -> Tuple[bool, Dict]:
		cur_pos, cur_orn = p.getLinkState(self.arm, self.END_IDX)[:2]
		
		pos_delta = np.clip(action[:3], -self.MAX_POS_STEP, self.MAX_POS_STEP)
		target_pos = np.array(cur_pos) + pos_delta
		
		target_orn = cur_orn
		
		ik_joints = p.calculateInverseKinematics(
			self.arm, self.END_IDX, target_pos, targetOrientation=target_orn,
			restPoses=self.INIT_JOINTS + [self.GRIPPER_OPEN, self.GRIPPER_OPEN],
			jointDamping=[0.1] * self.NUM_MOVABLE_JOINTS
		)
		
		for i in range(self.NUM_ARM_JOINTS):
			p.setJointMotorControl2(self.arm, i, p.POSITION_CONTROL,
								   targetPosition=ik_joints[i], force=self.JOINT_FORCE)
		
		gripper_cmd = action[3]
		if gripper_cmd < 0:
			self.gripper_closed = True
		elif gripper_cmd > 0:
			self.gripper_closed = False
		gripper_pos = self.GRIPPER_CLOSED if self.gripper_closed else self.GRIPPER_OPEN
		
		for gj in self.GRIPPER_JOINTS:
			p.setJointMotorControl2(self.arm, gj, p.POSITION_CONTROL,
								   targetPosition=gripper_pos, force=self.GRIPPER_FORCE)
		
		p.stepSimulation()
		
		if self.render and self.debug:
			self._update_gripper_helpers()
		
		info = {
			'reached_height': self._check_height_reached(),
			'bnd_vio': self._check_bnd_vio(),
			'gripper_closed': self.gripper_closed
		}
		
		return (self._is_done(), info)
	
	def set_camera_view(self, view_name):
		if not self.render or view_name not in self.CAMERA_CONFIGS:
			return
		
		config = self.CAMERA_CONFIGS[view_name]
		p.resetDebugVisualizerCamera(config['distance'], config['yaw'], 
									 config['pitch'], config['target'])
		if self.verbose:
			print(f"Switched to {view_name} view")
	
	def close(self):
		p.disconnect()


def test_env(show_bnd=False, randomize=False, debug=False, seed=None):
	env = ArmEnv(render=True, verbose=False, debug=debug, show_bnd=show_bnd, randomize=randomize, seed=seed)
	current_view = 'front'
	
	print("\nEnvironment initialized!")
	print("=" * 50)
	print("7-DoF Arm (Franka Panda)")
	print("Red rod: target object")
	print("Red circle: boundary (1m radius)")
	print("=" * 50)
	print("\nTask: Lift the rod to the target height (0.5m)")
	print("Keep the rod within 1m radius circle from origin")
	print("\nKeyboard controls:")
	print("Arm movement (IK control):")
	print("  Z/C: move end-effector down/up")
	print("  Arrow keys: move end-effector left/right/forward/backward")
	print("\nGripper control:")
	print("  Space: close gripper")
	print("  B: open gripper")
	print("\nView switching:")
	print("  1: front view")
	print("  2: top view")
	print("  3: side view")
	print("\nOther:")
	print("  Ctrl+C: exit")
	print("=" * 50)
	print("Start simulation...")
	
	step = 0
	
	try:
		while True:
			keys = p.getKeyboardEvents()
			action = np.zeros(4)
			
			for key, event in keys.items():
				if event & p.KEY_IS_DOWN:
					if key in [ord('c'), ord('C')]:
						action[2] += 1.0
					elif key in [ord('z'), ord('Z')]:
						action[2] -= 1.0
					elif key == p.B3G_LEFT_ARROW:
						action[0] -= 1.0
					elif key == p.B3G_RIGHT_ARROW:
						action[0] += 1.0
					elif key == p.B3G_UP_ARROW:
						action[1] += 1.0
					elif key == p.B3G_DOWN_ARROW:
						action[1] -= 1.0
					elif key == p.B3G_SPACE:
						action[3] = -1.0
					elif key in [ord('b'), ord('B')]:
						action[3] = 1.0
					elif event & p.KEY_WAS_TRIGGERED:
						if key == ord('1'):
							current_view = 'front'
							env.set_camera_view(current_view)
						elif key == ord('2'):
							current_view = 'top'
							env.set_camera_view(current_view)
						elif key == ord('3'):
							current_view = 'side'
							env.set_camera_view(current_view)
			
			done, info = env.step(action)

			time.sleep(1./60.)
			step += 1
			
			if (step + 1) % 100 == 0:
				pos, _ = p.getBasePositionAndOrientation(env.rod)
				height = pos[2]
				dist_from_origin = np.linalg.norm(np.array(pos[:2]))
				st = f"reached:{info['reached_height']} bnd_vio:{info['bnd_vio']} grip:{info['gripper_closed']}"
				print(f"{step + 1:4d}: [{st}] height:{height:.3f}m dist_origin:{dist_from_origin:.3f}m")
			
			if done:
				msg = "Boundary violation" if info['bnd_vio'] else "Target height reached"
				print(f"\n{msg}! steps:{step + 1}")
				break
	
	except KeyboardInterrupt:
		print("\n\nExiting simulation")
	
	print(f"Total {step + 1} steps")
	env.close()


if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description='PRML Project - ArmEnv Test Environment')
	parser.add_argument('-d', '--debug', action='store_true', help='Show debug info')
	parser.add_argument('-b', '--show-boundary', action='store_true', help='Show boundary markers')
	parser.add_argument('-r', '--randomize', action='store_true', help='Randomize initial positions')
	parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed (default: random)')
	
	args = parser.parse_args()
	
	test_env(show_bnd=args.show_boundary, randomize=args.randomize, debug=args.debug, seed=args.seed)
