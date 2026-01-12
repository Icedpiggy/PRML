import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import Tuple, Dict, List


class SimpleRodEnv:
	ROD_L = 0.2  # Single rod length (m)
	ROD_R = 0.02  # Rod radius (m)
	ROD_M = 0.25  # Rod mass (kg)
	COMB_L = ROD_L * 2  # Combined rod length (m)
	COMB_M = ROD_M * 2  # Combined rod mass (kg)
	TGT_TH = 0.05  # Target hit threshold (m)
	TOL = 0.04  # Connection tolerance (m)
	END_TH = ROD_L * TOL  # End-to-end threshold for connection
	CTR_MIN = ROD_L * (1.0 - TOL)  # Min center distance for connection
	CTR_MAX = ROD_L * (1.0 + TOL)  # Max center distance for connection
	BND_R = 2.0  # Boundary radius (m)
	
	def __init__(self, render=True, verbose=False, debug=False, show_bnd=False, randomize=False, hard=False):
		self.render = render
		self.verbose = verbose
		self.show_bnd = show_bnd
		self.randomize = randomize
		self.hard = hard
		seed = int(time.time_ns())
		self.rng = np.random.default_rng(seed)
		if self.verbose:
			print(f"Random seed: {seed}")
		
		self.pc = p.connect(p.GUI if render else p.DIRECT)
		if render:
			p.resetDebugVisualizerCamera(1.5, 0, 0, [0.5, 0, 0.3])
			p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
		
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.81)
		p.loadURDF("plane.urdf")
		
		self.rod_a = self.rod_b = self.comb = None
		self.wall = None
		self.tgt_id = None
		self.tgt_pos = None
		self.conn = False
		
		if self.show_bnd:
			self._create_bnd_marker()
		
		self.reset()
	
	def _create_rod(self, color, pos, length=ROD_L, mass=ROD_M, orn=[0, 0, 0, 1]):
		v = p.createVisualShape(p.GEOM_CYLINDER, radius=self.ROD_R, length=length, rgbaColor=color)
		c = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.ROD_R, height=length)
		rod_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=c,
								   baseVisualShapeIndex=v, basePosition=pos, baseOrientation=orn)
		p.changeDynamics(rod_id, -1, lateralFriction=0.9, spinningFriction=0.1, rollingFriction=0.01)
		return rod_id
	
	def _create_walls(self):
		he = [0.5, 0.02, 0.5]
		wv = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.7, 0.7, 0.7, 1])
		wc = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
		
		if self.hard:
			wall_x, wall_y, wall_z = self.rng.uniform(-0.5, 0.5), self.rng.uniform(0.9, 1.05), 0.5
			wall_angle = self.rng.uniform(-np.pi/6, np.pi/6)
			wall_orn = p.getQuaternionFromEuler([0, 0, wall_angle])
			
			tgt_r = self.rng.uniform(0.0, 0.25)
			tgt_a = self.rng.uniform(0, 2 * np.pi)
			tgt_local = [tgt_r * np.cos(tgt_a), 0, tgt_r * np.sin(tgt_a)]
			self.tgt_pos = [wall_x + tgt_local[0] * np.cos(wall_angle) - tgt_local[1] * np.sin(wall_angle),
							wall_y + tgt_local[0] * np.sin(wall_angle) + tgt_local[1] * np.cos(wall_angle),
							wall_z + tgt_local[2]]
		else:
			wall_x, wall_y, wall_z = 0.0, 1.0, 0.5
			wall_orn = [0, 0, 0, 1]
			self.tgt_pos = [wall_x, wall_y, wall_z]
		
		self.wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wc,
									 baseVisualShapeIndex=wv,
									 basePosition=[wall_x, wall_y, wall_z],
									 baseOrientation=wall_orn)
		
		tv = p.createVisualShape(p.GEOM_SPHERE, radius=self.TGT_TH, rgbaColor=[1, 1, 0, 1])
		self.tgt_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=tv, basePosition=self.tgt_pos)
		
		return self.wall, self.tgt_id
	
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
		def is_out(rid):
			pos = p.getBasePositionAndOrientation(rid)[0]
			return np.linalg.norm(np.array(pos[:2])) > self.BND_R
		if self.conn and self.comb:
			return is_out(self.comb)
		return is_out(self.rod_a) or is_out(self.rod_b)
	
	def _get_axis(self, q):
		mat = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
		return mat @ np.array([0, 0, 1])
	
	def _get_ends(self, rid, length=ROD_L) -> List[np.ndarray]:
		pos, orn = p.getBasePositionAndOrientation(rid)
		axis = self._get_axis(orn)
		return [np.array(pos) + axis * (length / 2), np.array(pos) - axis * (length / 2)]
	
	def _rand_pos(self, x_rng, y_rng, z):
		return [self.rng.uniform(*x_rng), self.rng.uniform(*y_rng), z]
	
	def _rand_orn(self):
		if self.hard and self.rng.random() < 0.5:
			axis_angle = self.rng.uniform(0, 2 * np.pi)
			return p.getQuaternionFromEuler([np.pi/2, 0, axis_angle])
		else:
			return [0, 0, 0, 1]
	
	def _get_orn_type(self, orn):
		mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
		axis = mat @ np.array([0, 0, 1])
		z_alignment = abs(axis[2])
		return 'upright' if z_alignment > 0.9 else 'flat'
	
	def reset(self):
		if self.comb:
			p.removeBody(self.comb)
			self.comb = None
		self.conn = False
		
		if self.rod_a is not None:
			p.removeBody(self.rod_a)
		if self.rod_b is not None:
			p.removeBody(self.rod_b)
		
		orn_a = self._rand_orn() if self.randomize else [0, 0, 0, 1]
		orn_b = self._rand_orn() if self.randomize else [0, 0, 0, 1]
		
		def get_pos_z(orn):
			orn_type = self._get_orn_type(orn)
			if orn_type == 'flat':
				return self.ROD_R
			else:
				return self.ROD_L / 2
		
		if self.randomize:
			pos_a = [self.rng.uniform(0.1, 0.4), self.rng.uniform(0.2, 0.4), get_pos_z(orn_a)]
			pos_b = [self.rng.uniform(-0.4, -0.1), self.rng.uniform(0.2, 0.4), get_pos_z(orn_b)]
		else:
			pos_a = [0.4, 0.2, get_pos_z(orn_a)]
			pos_b = [-0.4, 0.2, get_pos_z(orn_b)]
		
		self.rod_a = self._create_rod([1, 0, 0, 1], pos_a, orn=orn_a)
		self.rod_b = self._create_rod([0, 0, 1, 1], pos_b, orn=orn_b)
		
		if self.wall is not None:
			p.removeBody(self.wall)
		if self.tgt_id is not None:
			p.removeBody(self.tgt_id)
		
		self._create_walls()
		
		for rid, pos, orn in [(self.rod_a, pos_a, orn_a), (self.rod_b, pos_b, orn_b)]:
			p.resetBasePositionAndOrientation(rid, pos, orn)
			p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
		
		return self.get_obs()
	
	def get_obs(self) -> np.ndarray:
		if self.conn and self.comb:
			pos, orn = p.getBasePositionAndOrientation(self.comb)
			vel, ang_v = p.getBaseVelocity(self.comb)
			return np.array(list(pos) + list(orn) + list(vel) + list(ang_v) +
						  list(pos) + list(orn) + list(vel) + list(ang_v) + list(self.tgt_pos))
		
		pos_a, orn_a = p.getBasePositionAndOrientation(self.rod_a)
		vel_a, ang_a = p.getBaseVelocity(self.rod_a)
		pos_b, orn_b = p.getBasePositionAndOrientation(self.rod_b)
		vel_b, ang_b = p.getBaseVelocity(self.rod_b)
		return np.array(list(pos_a) + list(orn_a) + list(vel_a) + list(ang_a) +
						list(pos_b) + list(orn_b) + list(vel_b) + list(ang_b) + list(self.tgt_pos))
	
	def step(self, action: np.ndarray) -> Tuple[bool, Dict]:
		if self.conn and self.comb:
			force = [sum(action[i::3]) * 10 for i in range(3)]
			p.applyExternalForce(self.comb, -1, force, [0, 0, 0], p.WORLD_FRAME)
		else:
			for rid, off in [(self.rod_a, 0), (self.rod_b, 3)]:
				f = [action[i + off] * 10 for i in range(3)]
				p.applyExternalForce(rid, -1, f, [0, 0, 0], p.WORLD_FRAME)
		
		p.stepSimulation()
		
		return (self._is_done(),
				{'conn': self._check_conn(), 'hit': self._check_hit(), 'bnd_vio': self._check_bnd_vio()})
	
	def _check_conn(self) -> bool:
		if self.conn:
			return True
		
		pos_a, orn_a = p.getBasePositionAndOrientation(self.rod_a)
		pos_b, orn_b = p.getBasePositionAndOrientation(self.rod_b)
		
		ends_a, ends_b = self._get_ends(self.rod_a), self._get_ends(self.rod_b)
		min_d = min(np.linalg.norm(pa - pb) for pa in ends_a for pb in ends_b)
		ctr_d = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
		
		if min_d < self.END_TH and self.CTR_MIN <= ctr_d <= self.CTR_MAX:
			all_ends = ends_a + ends_b
			mp = max(((i, j, np.linalg.norm(all_ends[i] - all_ends[j]))
					 for i in range(len(all_ends)) for j in range(i+1, len(all_ends))), key=lambda x: x[2])
			
			end1, end2 = all_ends[mp[0]], all_ends[mp[1]]
			cp = (end1 + end2) / 2
			direction = end1 - end2
			direction = direction / np.linalg.norm(direction)
			
			z_axis = np.array([0, 0, 1])
			cross = np.cross(z_axis, direction)
			dot = np.dot(z_axis, direction)
			cross_norm = np.linalg.norm(cross)
			
			if cross_norm < 1e-6:
				orn = [0, 0, 0, 1] if dot > 0 else [1, 0, 0, 0]
			else:
				sin_half = cross_norm / 2
				cos_half = (1 + dot) / 2
				norm = np.sqrt(sin_half**2 + cos_half**2)
				orn = [cross[0]/(2*norm), cross[1]/(2*norm), cross[2]/(2*norm), cos_half/norm]
			
			p.removeBody(self.rod_a)
			p.removeBody(self.rod_b)
			self.rod_a = self.rod_b = None
			
			self.comb = self._create_rod([0.5, 0, 0.5, 1], cp, self.COMB_L, self.COMB_M, orn)
			p.resetBaseVelocity(self.comb, [0, 0, 0], [0, 0, 0])
			
			self.conn = True
			if self.verbose:
				axis_a, axis_b = self._get_axis(orn_a), self._get_axis(orn_b)
				ang_d = np.degrees(np.arccos(min(abs(np.dot(axis_a, axis_b)), 1.0)))
				print(f"\nMerged! dist={min_d:.4f}m, angle={ang_d:.1f}°")
			
			return True
		return False
	
	def _check_hit(self) -> bool:
		if self.conn and self.comb:
			ends = self._get_ends(self.comb, self.COMB_L)
			tgt = np.array(self.tgt_pos)
			return min(np.linalg.norm(e - tgt) for e in ends) < self.TGT_TH
		return np.linalg.norm(np.array(p.getBasePositionAndOrientation(self.rod_a)[0]) -
							np.array(self.tgt_pos)) < self.TGT_TH
	
	def _is_done(self) -> bool:
		return (self.conn and self._check_hit()) or self._check_bnd_vio()
	
	def close(self):
		p.disconnect()


def test_env(show_bnd=False, randomize=False, hard=False):
	print(f"Init env... (bnd:{'on' if show_bnd else 'off'}, rand:{'on' if randomize else 'off'}, hard:{'on' if hard else 'off'})")
	env = SimpleRodEnv(render=True, verbose=False, debug=True, show_bnd=show_bnd, randomize=randomize, hard=hard)
	step = 0
	print("Start simulation...")
	
	while True:
		done, info = env.step(np.zeros(6))
		
		if (step + 1) % 100 == 0:
			st = f"conn:{info['conn']} hit:{info['hit']}"
			
			if env.conn and env.comb:
				ends = env._get_ends(env.comb, env.COMB_L)
				tgt = np.array(env.tgt_pos)
				d = min(np.linalg.norm(e - tgt) for e in ends)
				print(f"{step + 1:4d}: [{st}] end_tgt_dist:{d:.3f}m")
			else:
				ends_a, ends_b = env._get_ends(env.rod_a), env._get_ends(env.rod_b)
				end_d = min(np.linalg.norm(pa - pb) for pa in ends_a for pb in ends_b)
				pos_a, pos_b = p.getBasePositionAndOrientation(env.rod_a)[0], p.getBasePositionAndOrientation(env.rod_b)[0]
				ctr_d = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
				print(f"{step + 1:4d}: [{st}] end_dist:{end_d:.3f}m ctr_dist:{ctr_d:.3f}m")
		
		if done:
			msg = "Boundary violation" if info['bnd_vio'] else "Task complete"
			print(f"\n{msg}! steps:{step + 1}")
			break
		step += 1
		time.sleep(1./60.)
	
	print(f"Total {step + 1} steps")
	env.close()


if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description='PRML Project - SimpleRodEnv Test Environment')
	parser.add_argument('-b', '--show-boundary', action='store_true', help='Show boundary markers')
	parser.add_argument('-r', '--randomize', action='store_true', help='Randomize initial positions')
	parser.add_argument('--hard', action='store_true', help='Enable hard mode (rods may be flat)')
	
	args = parser.parse_args()
	
	test_env(show_bnd=args.show_boundary, randomize=args.randomize, hard=args.hard)
