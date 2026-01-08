import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import Tuple, Dict, List


class SimpleRodEnv:
    ROD_L = 0.2  # Single rod length (m)
    ROD_R = 0.02  # Rod radius (m)
    ROD_M = 0.5  # Rod mass (kg)
    COMB_L = ROD_L * 2  # Combined rod length (m)
    COMB_M = ROD_M * 2  # Combined rod mass (kg)
    TGT_TH = 0.08  # Target hit threshold (m)
    TOL = 0.04  # Connection tolerance (m)
    END_TH = ROD_L * TOL  # End-to-end threshold for connection
    CTR_MIN = ROD_L * (1.0 - TOL)  # Min center distance for connection
    CTR_MAX = ROD_L * (1.0 + TOL)  # Max center distance for connection
    BND_R = 2.0  # Boundary radius (m)
    
    def __init__(self, render=True, verbose=False, debug=False, show_bnd=False, randomize=False):
        self.render = render
        self.verbose = verbose
        self.show_bnd = show_bnd
        self.randomize = randomize
        self.rng = np.random.RandomState()
        
        self.pc = p.connect(p.GUI if render else p.DIRECT)
        if render:
            p.resetDebugVisualizerCamera(1.5, 0, 0, [0.5, 0, 0.3])
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        self.rod_a = self.rod_b = self.comb = None
        self.wall = None
        self.tgt_pos = None
        self.conn = False
        self.reset()
    
    def _create_rod(self, color, pos, length=ROD_L, mass=ROD_M, orn=[0, 0, 0, 1]):
        v = p.createVisualShape(p.GEOM_CYLINDER, radius=self.ROD_R, length=length, rgbaColor=color)
        c = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.ROD_R, height=length)
        return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=c,
                                baseVisualShapeIndex=v, basePosition=pos, baseOrientation=orn)
    
    def _create_walls(self):
        he = [0.5, 0.02, 0.5]
        wv = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.7, 0.7, 0.7, 1])
        wc = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
        
        if self.randomize:
            wall_x = self.rng.uniform(-0.5, 0.5)
            wall_y = self.rng.uniform(0.7, 0.9)
            wall_z = 0.5
            angle = self.rng.uniform(-np.pi/12, np.pi/12)
            wall_orn = [0, 0, np.sin(angle/2), np.cos(angle/2)]
            self.tgt_pos = [wall_x - 0.03 * np.cos(angle), wall_y - 0.03 * np.sin(angle), wall_z]
        else:
            wall_x, wall_y, wall_z = 0.0, 0.8, 0.5
            wall_orn = [0, 0, 1, 0]
            self.tgt_pos = [wall_x - 0.03, wall_y, wall_z]
        
        self.wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wc,
                                     baseVisualShapeIndex=wv,
                                     basePosition=[wall_x, wall_y, wall_z],
                                     baseOrientation=wall_orn)
        
        tv = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 1, 0, 1])
        self.tgt_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=tv, basePosition=self.tgt_pos)
        
        if self.show_bnd:
            self._create_bnd_marker()
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
    
    def reset(self):
        if self.comb:
            p.removeBody(self.comb)
            self.comb = None
        self.conn = False
        
        if self.randomize:
            pos_a = self._rand_pos((-0.5, 0.5), (0.1, 0.4), 0.1)
            pos_b = self._rand_pos((-0.5, 0.5), (0.1, 0.4), 0.1)
        else:
            pos_a = [0.5, 0.2, 0.1]
            pos_b = [-0.5, 0.2, 0.1]
        
        if self.rod_a is None or self.rod_b is None:
            self.rod_a = self._create_rod([1, 0, 0, 1], pos_a)
            self.rod_b = self._create_rod([0, 0, 1, 1], pos_b)
        if self.wall is None:
            self._create_walls()
        
        for rid, pos in [(self.rod_a, pos_a), (self.rod_b, pos_b)]:
            p.resetBasePositionAndOrientation(rid, pos, [0, 0, 0, 1])
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
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.conn and self.comb:
            force = [sum(action[i::3]) * 10 for i in range(3)]
            p.applyExternalForce(self.comb, -1, force, [0, 0, 0], p.WORLD_FRAME)
        else:
            for rid, off in [(self.rod_a, 0), (self.rod_b, 3)]:
                f = [action[i + off] * 10 for i in range(3)]
                p.applyExternalForce(rid, -1, f, [0, 0, 0], p.WORLD_FRAME)
        
        p.stepSimulation()
        if self.render:
            time.sleep(1./60.)
        
        return (self.get_obs(), self._calc_reward(), self._is_done(),
                {'conn': self._check_conn(), 'hit': self._check_hit(), 'bnd_vio': self._check_bnd_vio()})
    
    def _calc_reward(self) -> float:
        r = 0.0
        
        if self.conn and self.comb:
            ends = self._get_ends(self.comb, self.COMB_L)
            tgt = np.array(self.tgt_pos)
            r += 1.0 - min(np.linalg.norm(e - tgt) for e in ends)
        else:
            pos_a, pos_b = p.getBasePositionAndOrientation(self.rod_a)[0], p.getBasePositionAndOrientation(self.rod_b)[0]
            avg = (np.array(pos_a) + np.array(pos_b)) / 2
            d = np.linalg.norm(avg - np.array(self.tgt_pos))
            r += 1.0 - d * 2
            
            ends_a, ends_b = self._get_ends(self.rod_a), self._get_ends(self.rod_b)
            end_d = min(np.linalg.norm(pa - pb) for pa in ends_a for pb in ends_b)
            ctr_d = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
            
            if end_d < self.END_TH * 10:
                r += (1.0 - end_d / (self.END_TH * 10)) * 5
            if abs(ctr_d - self.ROD_L) < 0.05:
                r += (1.0 - abs(ctr_d - self.ROD_L) / 0.05) * 5
        
        if self._check_conn():
            r += 10
        if self._check_hit():
            r += 50
        
        return r - 0.1
    
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


def test_env(show_bnd=False, randomize=False):
    print(f"Init env... (bnd:{'on' if show_bnd else 'off'}, rand:{'on' if randomize else 'off'})")
    env = SimpleRodEnv(render=True, verbose=False, debug=True, show_bnd=show_bnd, randomize=randomize)
    step = 0
    print("Start simulation...")
    
    while True:
        obs, reward, done, info = env.step(np.zeros(6))
        
        if (step + 1) % 100 == 0:
            st = f"conn:{info['conn']} hit:{info['hit']}"
            
            if env.conn and env.comb:
                ends = env._get_ends(env.comb, env.COMB_L)
                tgt = np.array(env.tgt_pos)
                d = min(np.linalg.norm(e - tgt) for e in ends)
                print(f"{step + 1:4d}: [{st}] end_tgt_dist:{d:.3f}m reward:{reward:.2f}")
            else:
                ends_a, ends_b = env._get_ends(env.rod_a), env._get_ends(env.rod_b)
                end_d = min(np.linalg.norm(pa - pb) for pa in ends_a for pb in ends_b)
                pos_a, pos_b = p.getBasePositionAndOrientation(env.rod_a)[0], p.getBasePositionAndOrientation(env.rod_b)[0]
                ctr_d = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
                print(f"{step + 1:4d}: [{st}] end_dist:{end_d:.3f}m ctr_dist:{ctr_d:.3f}m reward:{reward:.2f}")
        
        if done:
            msg = "Boundary violation" if info['bnd_vio'] else "Task complete"
            print(f"\n{msg}! steps:{step + 1} reward:{reward:.2f}")
            break
        step += 1
    
    print(f"Total {step + 1} steps")
    env.close()


if __name__ == "__main__":
    import sys
    show_bnd = '--show-boundary' in sys.argv or '-b' in sys.argv
    randomize = '--randomize' in sys.argv or '-r' in sys.argv
    test_env(show_bnd=show_bnd, randomize=randomize)
