"""Measure foot contact, clearance and tilt for the balance demo."""
import math
import mujoco
import numpy as np

SIDE_COMMAND = {'left': -1.0, 'right': 1.0}


class FootMeasurements:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.floor = next(
            i for i, kind in enumerate(model.geom_type) if kind == mujoco.mjtGeom.mjGEOM_PLANE
        )
        self.feet = {}
        for side in SIDE_COMMAND:
            geom = model.geom(f"robot/{side}_foot_collision").id
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            stop = start + model.mesh_vertnum[mesh]
            self.feet[side] = geom, model.mesh_vert[start:stop]

    def sample(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        touching = set()
        for contact in self.data.contact[:self.data.ncon]:
            first, second = map(int, contact.geom)
            if first == self.floor:
                touching.add(second)
            elif second == self.floor:
                touching.add(first)
        row = {}
        for side, (geom, vertices) in self.feet.items():
            world_z = self.data.geom_xpos[geom, 2] + vertices @ self.data.geom_xmat[geom, 6:9]
            row[f"{side}_clearance_m"] = float(world_z.min() - self.data.geom_xpos[self.floor, 2])
            row[f"{side}_contact"] = geom in touching
        row["root_height_m"] = float(qpos[2])
        row["tilt_deg"] = math.degrees(math.acos(np.clip(1 - 2 * (qpos[4]**2 + qpos[5]**2), -1, 1)))
        return row
