import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import Tuple, Dict


class SimpleRodEnv:
	ROD_L = 0.2  # Rod length (m)
	ROD_R = 0.02  # Rod radius (m)
	ROD_M = 0.25  # Rod mass (kg)
	TGT_HEIGHT = 0.5  # Target height (m) - rod center should reach this height
	BND_R = 1.0  # Boundary radius (m)
	
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
		if self.verbose:
			print(f"Random seed: {seed}")
		
		self.pc = p.connect(p.GUI if render else p.DIRECT)
		p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

		if render:
			p.resetDebugVisualizerCamera(1.5, 0, 0, [0.5, 0, 0.3])
			p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
		
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.81)
		p.loadURDF("plane.urdf")
		
		self.rod = None
		self.tgt_id = None
		
		if self.show_bnd:
			self._create_bnd_marker()
		
		if self.render and self.debug:
			self._create_target_marker()
		
		self.reset()
	
	def _create_rod(self, color, pos, length=ROD_L, mass=ROD_M, orn=[0, 0, 0, 1]):
		v = p.createVisualShape(p.GEOM_CYLINDER, radius=self.ROD_R, length=length, rgbaColor=color)
		c = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.ROD_R, height=length)
		rod_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=c,
								   baseVisualShapeIndex=v, basePosition=pos, baseOrientation=orn)
		p.changeDynamics(rod_id, -1, lateralFriction=0.9, spinningFriction=0.1, rollingFriction=0.01)
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
	
	def _create_target_marker(self):
		# Create a translucent plane at target height
		tv = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.BND_R, self.BND_R, 0.01], 
							   rgbaColor=[1, 1, 0, 0.3])
		self.tgt_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=tv, 
									  basePosition=[0, 0, self.TGT_HEIGHT], baseOrientation=[0, 0, 0, 1])
	
	def _check_bnd_vio(self) -> bool:
		pos = p.getBasePositionAndOrientation(self.rod)[0]
		return np.linalg.norm(np.array(pos[:2])) > self.BND_R
	
	def _get_axis(self, q):
		mat = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
		return mat @ np.array([0, 0, 1])
	
	def _get_ends(self, rid, length=ROD_L):
		pos, orn = p.getBasePositionAndOrientation(rid)
		axis = self._get_axis(orn)
		return [np.array(pos) + axis * (length / 2), np.array(pos) - axis * (length / 2)]
	
	def _rand_pos(self, x_rng, y_rng, z):
		return [self.rng.uniform(*x_rng), self.rng.uniform(*y_rng), z]
	
	def _rand_orn(self):
		return [0, 0, 0, 1]  # Default upright orientation
	
	def reset(self):
		if self.rod is not None:
			p.removeBody(self.rod)
		
		orn = self._rand_orn() if self.randomize else [0, 0, 0, 1]
		
		if self.randomize:
			pos = [self.rng.uniform(-0.4, 0.4), self.rng.uniform(0.2, 0.4), self.ROD_L / 2]
		else:
			pos = [0.0, 0.3, self.ROD_L / 2]
		
		self.rod = self._create_rod([1, 0, 0, 1], pos, orn=orn)
		
		return self.get_obs()
	
	def get_obs(self) -> np.ndarray:
		pos, orn = p.getBasePositionAndOrientation(self.rod)
		vel, ang_v = p.getBaseVelocity(self.rod)
		return np.array(list(pos) + list(orn) + list(vel) + list(ang_v))
	
	def step(self, action: np.ndarray) -> Tuple[bool, Dict]:
		force = [action[i] * 10 for i in range(3)]
		torque = [action[i] * 0.5 for i in range(3, 6)]
		p.applyExternalForce(self.rod, -1, force, [0, 0, 0], p.WORLD_FRAME)
		p.applyExternalTorque(self.rod, -1, torque, p.WORLD_FRAME)
		
		p.stepSimulation()
		
		return (self._is_done(),
				{'reached_height': self._check_height_reached(), 'bnd_vio': self._check_bnd_vio()})
	
	def _check_height_reached(self) -> bool:
		pos, _ = p.getBasePositionAndOrientation(self.rod)
		return pos[2] > self.TGT_HEIGHT
	
	def _is_done(self) -> bool:
		return self._check_height_reached() or self._check_bnd_vio()
	
	def close(self):
		p.disconnect()


def test_env(show_bnd=False, randomize=False, debug=False, seed=None):
	print(f"Init env... (bnd:{'on' if show_bnd else 'off'}, rand:{'on' if randomize else 'off'}, debug:{'on' if debug else 'off'}, seed:{seed})")
	env = SimpleRodEnv(render=True, verbose=False, debug=debug, show_bnd=show_bnd, randomize=randomize, seed=seed)
	step = 0
	print("Start simulation...")
	
	while True:
		action = np.zeros(6)
		done, info = env.step(action)
		
		if (step + 1) % 100 == 0:
			pos, _ = p.getBasePositionAndOrientation(env.rod)
			height = pos[2]
			dist_from_origin = np.linalg.norm(np.array(pos[:2]))
			st = f"height_reached:{info['reached_height']} bnd_vio:{info['bnd_vio']}"
			print(f"{step + 1:4d}: [{st}] height:{height:.3f}m dist_origin:{dist_from_origin:.3f}m")
		
		if done:
			msg = "Boundary violation" if info['bnd_vio'] else "Target height reached"
			print(f"\n{msg}! steps:{step + 1}")
			break
		step += 1
		time.sleep(1./60.)
	
	print(f"Total {step + 1} steps")
	env.close()


if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description='PRML Project - SimpleRodEnv Test Environment')
	parser.add_argument('-d', '--debug', action='store_true', help='Show debug info')
	parser.add_argument('-b', '--show-boundary', action='store_true', help='Show boundary markers')
	parser.add_argument('-r', '--randomize', action='store_true', help='Randomize initial positions')
	parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed (default: random)')
	
	args = parser.parse_args()
	
	test_env(show_bnd=args.show_boundary, randomize=args.randomize, debug=args.debug, seed=args.seed)
