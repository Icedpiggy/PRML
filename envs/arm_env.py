import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import Tuple, Dict, List


class ArmEnv:
	ROD_L = 0.2  # Single rod length (m)
	ROD_R = 0.02  # Rod radius (m)
	ROD_M = 0.25  # Rod mass (kg)
	COMB_L = ROD_L * 2  # Combined rod length (m)
	COMB_M = ROD_M * 2  # Combined rod mass (kg)
	TGT_TH = 0.05  # Target hit threshold (m)
	TOL = 0.1  # Connection tolerance
	END_TH = ROD_L * TOL  # End-to-end threshold for connection
	CTR_MIN = ROD_L * (1.0 - TOL)  # Min center distance for connection
	CTR_MAX = ROD_L * (1.0 + TOL)  # Max center distance for connection
	BND_R = 2.0  # Boundary radius (m)
	
	NUM_ARM_JOINTS = 7  # Number of arm joints
	NUM_MOVABLE_JOINTS = 9  # Number of movable joints (arm + gripper)
	END_IDX = 11  # End-effector link index
	GRIPPER_JOINTS = [9, 10]  # Gripper joint indices
	INIT_JOINTS = [0, -0.785, 0, -2.356, 0, 1.571, 0]  # Initial joint positions
	GRIPPER_OPEN = 0.04  # Gripper open position
	GRIPPER_CLOSED = 0.0  # Gripper closed position
	
	MAX_POS_STEP = 0.02  # Max position step per action
	MAX_ROT_STEP = 0.1  # Max rotation step per action
	JOINT_FORCE = 500  # Joint motor force
	GRIPPER_FORCE = 300  # Gripper motor force
	DT = 1./60.  # Simulation timestep
	
	CAMERA_CONFIGS = {
		'front': {'distance': 1.5, 'yaw': 0, 'pitch': -30, 'target': [0, 0, 0.3]},
		'top': {'distance': 1.5, 'yaw': 0, 'pitch': -89.9, 'target': [0, 0, 0.3]},
		'side': {'distance': 1.5, 'yaw': 90, 'pitch': -30, 'target': [0, 0, 0.3]},
	}  # Camera view configurations
	
	ROD_COLORS = {'rod_a': [1, 0, 0], 'rod_b': [0, 0, 1], 'comb': [0.5, 0, 0.5]}  # Rod colors
	DEBUG_LINE_LEN = 0.15  # Debug line extension length
	
	def __init__(self, render=True, verbose=False, debug=False, show_bnd=False, randomize=False, hard=False):
		self.render = render
		self.verbose = verbose
		self.debug = debug
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
			p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
		
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.81)
		p.loadURDF("plane.urdf")
		
		self.arm = None
		self.rod_a = self.rod_b = self.comb = None
		self.wall = None
		self.wall_pos = None
		self.wall_orn = None
		self.tgt_id = None
		self.tgt_pos = None
		self.bnd_markers = []
		self.conn = False
		self.gripper_closed = False
		self.gripper_helpers = {}
		self.rod_helpers = {}
		
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
	
	def _create_rod(self, color, pos, length=ROD_L, mass=ROD_M, orn=[0, 0, 0, 1]):
		v = p.createVisualShape(p.GEOM_CYLINDER, radius=self.ROD_R, length=length, rgbaColor=color)
		c = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.ROD_R, height=length)
		rod_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=c,
								   baseVisualShapeIndex=v, basePosition=pos, baseOrientation=orn)
		p.changeDynamics(rod_id, -1, lateralFriction=0.9, spinningFriction=0.1, rollingFriction=0.01,
						linearDamping=0.04, angularDamping=0.04, restitution=0.1)
		return rod_id
	
	def _add_rod_helpers(self, rod_id, rod_name, length=None):
		if not self.render or not self.debug:
			return
		
		color = self.ROD_COLORS.get(rod_name, [0, 0.6, 0])
		pos, orn = p.getBasePositionAndOrientation(rod_id)
		axis = self._get_axis(orn)
		end1 = np.array(pos) + axis * ((length or self.ROD_L) / 2)
		end2 = np.array(pos) - axis * ((length or self.ROD_L) / 2)
		
		self.rod_helpers[rod_name] = {
			'axis': p.addUserDebugLine(end2 - axis * self.DEBUG_LINE_LEN, end1 + axis * self.DEBUG_LINE_LEN, 
									   color, lineWidth=3, lifeTime=0)
		}
	
	def _update_rod_helpers(self):
		if not self.render or not self.debug or not self.rod_helpers:
			return
		
		for rod_id, rod_name, length in [(self.rod_a, 'rod_a', self.ROD_L), 
										 (self.rod_b, 'rod_b', self.ROD_L), 
										 (self.comb, 'comb', self.COMB_L)]:
			if rod_id is not None and rod_name in self.rod_helpers:
				self._update_single_rod_helper(rod_id, rod_name, length)
	
	def _update_single_rod_helper(self, rod_id, rod_name, length=None):
		if rod_name not in self.rod_helpers:
			return
		
		color = self.ROD_COLORS.get(rod_name, [0, 0.6, 0])
		length = length or self.ROD_L
		pos, orn = p.getBasePositionAndOrientation(rod_id)
		axis = self._get_axis(orn)
		end1 = np.array(pos) + axis * (length / 2)
		end2 = np.array(pos) - axis * (length / 2)
		
		p.addUserDebugLine(end2 - axis * self.DEBUG_LINE_LEN, end1 + axis * self.DEBUG_LINE_LEN, 
						  color, lineWidth=3, lifeTime=0, 
						  replaceItemUniqueId=self.rod_helpers[rod_name]['axis'])
	
	def _get_rod_state(self, rid):
		if rid is None:
			return [0.0] * 13
		pos, orn = p.getBasePositionAndOrientation(rid)
		vel, ang_v = p.getBaseVelocity(rid)
		return list(pos) + list(orn) + list(vel) + list(ang_v)
	
	def _create_walls(self):
		he = [0.5, 0.02, 0.5]
		wv = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.7, 0.7, 0.7, 1])
		wc = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
		
		if self.hard:
			wall_x, wall_y, wall_z = self.rng.uniform(-0.3, 0.3), self.rng.uniform(0.9, 1.0), 0.5
			wall_angle = self.rng.uniform(-np.pi/6, np.pi/6)
			wall_orn = p.getQuaternionFromEuler([0, 0, wall_angle])
			self.wall_orn = wall_orn
			
			tgt_r = self.rng.uniform(0.0, 0.2)
			tgt_a = self.rng.uniform(0, 2 * np.pi)
			tgt_local = [tgt_r * np.cos(tgt_a), 0, tgt_r * np.sin(tgt_a)]
			self.tgt_pos = [wall_x + tgt_local[0] * np.cos(wall_angle) - tgt_local[1] * np.sin(wall_angle),
							wall_y + tgt_local[0] * np.sin(wall_angle) + tgt_local[1] * np.cos(wall_angle),
							wall_z + tgt_local[2]]
		else:
			wall_x, wall_y, wall_z = 0.0, 1.0, 0.5
			wall_orn = [0, 0, 0, 1]
			self.wall_orn = wall_orn
			
			tgt_r = self.rng.uniform(0.0, 0.2)
			tgt_a = self.rng.uniform(0, 2 * np.pi)
			tgt_local = [tgt_r * np.cos(tgt_a), 0, tgt_r * np.sin(tgt_a)]
			self.tgt_pos = [wall_x + tgt_local[0], 
							wall_y + tgt_local[1], 
							wall_z + tgt_local[2]]
		
		self.wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wc,
									 baseVisualShapeIndex=wv,
									 basePosition=[wall_x, wall_y, wall_z],
									 baseOrientation=wall_orn)
		self.wall_pos = [wall_x, wall_y, wall_z]
		
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
			if rid is None:
				return False
			return np.linalg.norm(np.array(p.getBasePositionAndOrientation(rid)[0][:2])) > self.BND_R
		if self.conn and self.comb:
			return is_out(self.comb)
		return is_out(self.rod_a) or is_out(self.rod_b)
	
	def _get_axis(self, q):
		return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3) @ np.array([0, 0, 1])
	
	def _get_ends(self, rid, length=ROD_L) -> List[np.ndarray]:
		if rid is None:
			return [np.zeros(3), np.zeros(3)]
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
		if self.arm is None:
			self._create_arm()
		
		if self.comb:
			p.resetBasePositionAndOrientation(self.comb, [0, 0, 10], [0, 0, 0, 1])
			p.removeBody(self.comb)
			self.comb = None
		if self.rod_a is not None:
			p.resetBasePositionAndOrientation(self.rod_a, [0, 0, 10], [0, 0, 0, 1])
			p.removeBody(self.rod_a)
		if self.rod_b is not None:
			p.resetBasePositionAndOrientation(self.rod_b, [0, 0, 10], [0, 0, 0, 1])
			p.removeBody(self.rod_b)
		
		self.conn = False
		self.gripper_closed = False
		
		for i in range(self.NUM_ARM_JOINTS):
			p.resetJointState(self.arm, i, self.INIT_JOINTS[i], 0)
			p.setJointMotorControl2(self.arm, i, p.POSITION_CONTROL, targetPosition=self.INIT_JOINTS[i], force=self.JOINT_FORCE)
		
		for gj in self.GRIPPER_JOINTS:
			p.resetJointState(self.arm, gj, self.GRIPPER_OPEN, 0)
			p.setJointMotorControl2(self.arm, gj, p.POSITION_CONTROL, targetPosition=self.GRIPPER_OPEN, force=self.GRIPPER_FORCE)
		
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
		
		p.removeAllUserDebugItems()
		self.rod_helpers = {}
		self.gripper_helpers = {}
		
		if self.render and self.debug:
			for rid, name in [(self.rod_a, 'rod_a'), (self.rod_b, 'rod_b')]:
				self._add_rod_helpers(rid, name)
			self._add_gripper_helpers()
		
		if self.wall is not None:
			p.removeBody(self.wall)
		if self.tgt_id is not None:
			p.removeBody(self.tgt_id)
		
		self._create_walls()
		
		return self.get_obs()
	
	def get_obs(self) -> np.ndarray:
		joint_indices = list(range(self.NUM_ARM_JOINTS)) + self.GRIPPER_JOINTS
		joint_states = p.getJointStates(self.arm, joint_indices)
		joint_pos = [s[0] for s in joint_states]
		joint_vel = [s[1] for s in joint_states]
		
		end_pos, end_orn = p.getLinkState(self.arm, self.END_IDX)[:2]
		
		conn_state = [1.0] if self.conn else [0.0]
		
		rod_a_state = self._get_rod_state(self.rod_a)
		rod_b_state = self._get_rod_state(self.rod_b)
		rod_c_state = self._get_rod_state(self.comb)
		
		return np.array(conn_state + joint_pos + joint_vel + list(end_pos) + list(end_orn) +
					   rod_a_state + rod_b_state + rod_c_state + list(self.tgt_pos))
	
	def step(self, action: np.ndarray) -> Tuple[bool, Dict]:
		cur_pos, cur_orn = p.getLinkState(self.arm, self.END_IDX)[:2]
		
		pos_delta = np.clip(action[:3], -self.MAX_POS_STEP, self.MAX_POS_STEP)
		target_pos = np.array(cur_pos) + pos_delta
		
		rot_delta = np.clip(action[3:6], -self.MAX_ROT_STEP, self.MAX_ROT_STEP)
		target_euler = np.array(p.getEulerFromQuaternion(cur_orn)) + rot_delta
		target_orn = p.getQuaternionFromEuler(target_euler)
		
		ik_joints = p.calculateInverseKinematics(
			self.arm, self.END_IDX, target_pos, targetOrientation=target_orn,
			restPoses=self.INIT_JOINTS + [self.GRIPPER_OPEN, self.GRIPPER_OPEN],
			jointDamping=[0.1] * self.NUM_MOVABLE_JOINTS
		)
		
		for i in range(self.NUM_ARM_JOINTS):
			p.setJointMotorControl2(self.arm, i, p.POSITION_CONTROL,
								   targetPosition=ik_joints[i], force=self.JOINT_FORCE)
		
		gripper_cmd = action[6]
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
			self._update_rod_helpers()
		
		if self.render:
			time.sleep(self.DT)
		
		info = {
			'conn': self._check_conn(),
			'hit': self._check_hit(),
			'bnd_vio': self._check_bnd_vio()
		}
		
		return (self._is_done(), info)
	
	def _check_conn(self) -> bool:
		if self.conn:
			return True
		
		if self.rod_a is None or self.rod_b is None:
			return False
		
		try:
			pos_a, orn_a = p.getBasePositionAndOrientation(self.rod_a)
			pos_b, orn_b = p.getBasePositionAndOrientation(self.rod_b)
			
			ends_a, ends_b = self._get_ends(self.rod_a), self._get_ends(self.rod_b)
			min_d = min(np.linalg.norm(pa - pb) for pa in ends_a for pb in ends_b)
			ctr_d = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
		except Exception as e:
			if self.verbose:
				print(f"Error in _check_conn: {e}")
			return False
		
		if min_d < self.END_TH and self.CTR_MIN <= ctr_d <= self.CTR_MAX:
			try:
				all_ends = ends_a + ends_b
				mp = max(((i, j, np.linalg.norm(all_ends[i] - all_ends[j]))
						 for i in range(len(all_ends)) for j in range(i+1, len(all_ends))), key=lambda x: x[2])
				
				end1, end2 = all_ends[mp[0]], all_ends[mp[1]]
				cp = (end1 + end2) / 2
				direction = (end1 - end2) / np.linalg.norm(end1 - end2)
				
				axis_a, axis_b = self._get_axis(orn_a), self._get_axis(orn_b)
				ang_d = np.degrees(np.arccos(min(abs(np.dot(axis_a, axis_b)), 1.0)))
				
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
					
					if norm < 1e-10:
						if self.verbose:
							print("Warning: Invalid quaternion norm, using default orientation")
						orn = [0, 0, 0, 1]
					else:
						orn = [cross[0]/(2*norm), cross[1]/(2*norm), cross[2]/(2*norm), cos_half/norm]
					
					orn_norm = np.sqrt(sum(x**2 for x in orn))
					if orn_norm > 1e-10:
						orn = [x / orn_norm for x in orn]
					else:
						if self.verbose:
							print("Warning: Invalid quaternion, using default orientation")
						orn = [0, 0, 0, 1]
				
				for _ in range(3):
					try:
						p.stepSimulation()
					except Exception as e:
						if self.verbose:
							print(f"Error during simulation step: {e}")
				
				self.gripper_closed = False
				for gj in self.GRIPPER_JOINTS:
					try:
						p.resetJointState(self.arm, gj, self.GRIPPER_OPEN, 0)
						p.setJointMotorControl2(self.arm, gj, p.POSITION_CONTROL, 
											   targetPosition=self.GRIPPER_OPEN, force=self.GRIPPER_FORCE)
					except Exception as e:
						if self.verbose:
							print(f"Error setting gripper joint {gj}: {e}")
				
				try:
					p.setCollisionFilterPair(self.arm, self.rod_a, -1, -1, 0)
				except Exception as e:
					if self.verbose:
						print(f"Error setting collision filter for rod_a: {e}")
				try:
					p.setCollisionFilterPair(self.arm, self.rod_b, -1, -1, 0)
				except Exception as e:
					if self.verbose:
						print(f"Error setting collision filter for rod_b: {e}")
				
				try:
					p.resetBasePositionAndOrientation(self.rod_a, [0, 0, 10], [0, 0, 0, 1])
				except Exception as e:
					if self.verbose:
						print(f"Error moving rod_a away: {e}")
				try:
					p.resetBasePositionAndOrientation(self.rod_b, [0, 0, 10], [0, 0, 0, 1])
				except Exception as e:
					if self.verbose:
						print(f"Error moving rod_b away: {e}")
				
				for _ in range(3):
					try:
						p.stepSimulation()
					except Exception as e:
						if self.verbose:
							print(f"Error during simulation step (post-move): {e}")
				
				try:
					p.removeBody(self.rod_a)
				except Exception as e:
					if self.verbose:
						print(f"Error removing rod_a: {e}")
				try:
					p.removeBody(self.rod_b)
				except Exception as e:
					if self.verbose:
						print(f"Error removing rod_b: {e}")
				self.rod_a = self.rod_b = None
				
				try:
					p.removeAllUserDebugItems()
				except Exception as e:
					if self.verbose:
						print(f"Error removing debug items: {e}")
				self.rod_helpers = {}
				
				self.comb = self._create_rod([0.5, 0, 0.5, 1], cp, self.COMB_L, self.COMB_M, orn)
				try:
					p.resetBaseVelocity(self.comb, [0, 0, 0], [0, 0, 0])
				except Exception as e:
					if self.verbose:
						print(f"Error resetting combined rod velocity: {e}")
				try:
					p.setCollisionFilterPair(self.arm, self.comb, -1, -1, 0)
				except Exception as e:
					if self.verbose:
						print(f"Error setting collision filter for comb: {e}")
				
				for _ in range(5):
					try:
						p.stepSimulation()
					except Exception as e:
						if self.verbose:
							print(f"Error during final simulation step: {e}")
				
				if self.render and self.debug:
					self._add_gripper_helpers()
					self._add_rod_helpers(self.comb, 'comb', self.COMB_L)
				
				self.conn = True
				if self.verbose:
					print(f"\nMerged! dist={min_d:.4f}m, angle={ang_d:.1f}°")
				
				return True
			except Exception as e:
				if self.verbose:
					print(f"Error during rod merging: {e}")
				self.conn = True
				return True
		return False
	
	def _check_hit(self) -> bool:
		if self.conn and self.comb:
			ends = self._get_ends(self.comb, self.COMB_L)
			return min(np.linalg.norm(e - np.array(self.tgt_pos)) for e in ends) < self.TGT_TH
		if self.rod_a is None:
			return False
		return np.linalg.norm(np.array(p.getBasePositionAndOrientation(self.rod_a)[0]) -
							np.array(self.tgt_pos)) < self.TGT_TH
	
	def _is_done(self) -> bool:
		return (self.conn and self._check_hit()) or self._check_bnd_vio()
	
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


def test_env(show_bnd=False, randomize=False, debug=False, hard=False):
	env = ArmEnv(render=True, verbose=False, debug=debug, show_bnd=show_bnd, randomize=randomize, hard=hard)
	current_view = 'front'
	
	print("\nEnvironment initialized!")
	print("=" * 50)
	print("7-DoF Arm (Franka Panda)")
	print("Red rod: rod A")
	print("Blue rod: rod B")
	print("Purple rod: combined rod")
	print("Yellow sphere: target")
	print("=" * 50)
	print("\nKeyboard controls:")
	print("Arm movement (IK control):")
	print("  C/Z: move end-effector up/down")
	print("  Arrow keys: move end-effector left/right/forward/backward")
	print("\nGripper rotation:")
	print("  J/L: rotate around X axis (roll)")
	print("  I/K: rotate around Y axis (pitch)")
	print("  U/O: rotate around Z axis (yaw)")
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
			action = np.zeros(7)
			
			for key, event in keys.items():
				if event & p.KEY_IS_DOWN:
					if key in [ord('c'), ord('C')]:
						action[2] += 0.5
					elif key in [ord('z'), ord('Z')]:
						action[2] -= 0.5
					elif key == p.B3G_LEFT_ARROW:
						action[0] -= 0.5
					elif key == p.B3G_RIGHT_ARROW:
						action[0] += 0.5
					elif key == p.B3G_UP_ARROW:
						action[1] += 0.5
					elif key == p.B3G_DOWN_ARROW:
						action[1] -= 0.5
					elif key in [ord('j'), ord('J')]:
						action[3] += 0.5
					elif key in [ord('l'), ord('L')]:
						action[3] -= 0.5
					elif key in [ord('k'), ord('K')]:
						action[4] += 0.5
					elif key in [ord('i'), ord('I')]:
						action[4] -= 0.5
					elif key in [ord('u'), ord('U')]:
						action[5] += 0.5
					elif key in [ord('o'), ord('O')]:
						action[5] -= 0.5
					elif key == p.B3G_SPACE:
						action[6] = -1.0
					elif key in [ord('b'), ord('B')]:
						action[6] = 1.0
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
			step += 1
			
			if (step + 1) % 100 == 0:
				st = f"conn:{info['conn']} hit:{info['hit']}"
				
				if env.conn and env.comb:
					ends = env._get_ends(env.comb, env.COMB_L)
					d = min(np.linalg.norm(e - np.array(env.tgt_pos)) for e in ends)
					print(f"{step + 1:4d}: [{st}] end_tgt_dist:{d:.3f}m")
				else:
					if env.rod_a is not None and env.rod_b is not None:
						ends_a, ends_b = env._get_ends(env.rod_a), env._get_ends(env.rod_b)
						end_d = min(np.linalg.norm(pa - pb) for pa in ends_a for pb in ends_b)
						pos_a, pos_b = p.getBasePositionAndOrientation(env.rod_a)[0], p.getBasePositionAndOrientation(env.rod_b)[0]
						ctr_d = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
						print(f"{step + 1:4d}: [{st}] end_dist:{end_d:.3f}m ctr_dist:{ctr_d:.3f}m")
					else:
						print(f"{step + 1:4d}: [{st}] rods merged or unavailable")
			
			if done:
				msg = "Boundary violation" if info['bnd_vio'] else "Task complete"
				print(f"\n{msg}! steps:{step + 1}")
				break
	
	except KeyboardInterrupt:
		print("\n\nExiting simulation")
	
	print(f"Total {step + 1} steps")
	env.close()


if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description='PRML Project - ArmEnv Test Environment')
	parser.add_argument('-b', '--show-boundary', action='store_true', help='Show boundary markers')
	parser.add_argument('-r', '--randomize', action='store_true', help='Randomize initial positions')
	parser.add_argument('--hard', action='store_true', help='Enable hard mode (rods may be flat)')
	parser.add_argument('-d', '--debug', action='store_true', help='Show debug information')
	
	args = parser.parse_args()
	
	test_env(show_bnd=args.show_boundary, randomize=args.randomize, hard=args.hard, debug=args.debug)
