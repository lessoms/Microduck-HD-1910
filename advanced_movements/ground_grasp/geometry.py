"""Geometry adapted from pollen-robotics/microduck_rl @ 666dab6, robot/ground_grasp.py.
Upstream Apache-2.0; original source retained in teachers/static_grasp/ground_grasp_upstream.py.txt.
Only the local joint range is changed to the user's 5..35 degree specification.
Jaw actuator/contacts remain upstream approximations, not measured HD1910 jaw parameters.
"""
import mujoco
import numpy as np
MOUTH_JOINT = "passive_mouth"
MOUTH_PIVOT = np.array([0.0037, 0.0, -0.0182])
MOUTH_MIN_RAD = float(np.deg2rad(5.0))
MOUTH_MAX_RAD = float(np.deg2rad(35.0))
MOUTH_RANGE = (MOUTH_MIN_RAD, MOUTH_MAX_RAD)


def safe_mouth_target(angle_rad):
    # Single physical-angle boundary; fail closed on NaN/Inf rather than forwarding it.
    if not np.isfinite(angle_rad):
        raise ValueError("Nonfinite mouth angle")
    return float(np.clip(angle_rad, MOUTH_MIN_RAD, MOUTH_MAX_RAD))

JAW_MASS = 0.010
_UNDERSIDE_HULLS = ("bottom_head_shell", "soft_mouth_top", "jaw", "jaw_soft")
_JAW_MESHES = ("jaw", "jaw_soft")
OBJECT_HALF = (0.010, 0.010, 0.008)
OBJECT_MASS = 0.012


def _uf(u: float, f: float) -> np.ndarray:
    """Head-local position from (up, forward)."""
    return np.array([u, 0.0, -f])

def add_mouth(spec: mujoco.MjSpec, prefix: str = "") -> None:
    """Give the head an opening jaw and box collisions for the beak (see module docstring). Edits ``spec`` in place."""
    head = spec.body(prefix + "jaw_soft")
    jaw = head.add_body(name=prefix + "jaw", pos=MOUTH_PIVOT)
    jaw.mass = JAW_MASS
    jaw.inertia = [4e-6, 4e-6, 4e-6]
    jaw.ipos = _uf(-0.0134, 0.054) - MOUTH_PIVOT
    jaw.explicitinertial = True
    head.mass = head.mass - JAW_MASS
    jaw.add_joint(name=prefix + MOUTH_JOINT, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0],
                  range=list(MOUTH_RANGE), damping=0.02, armature=0.0018, frictionloss=0.005)
    for g in list(head.geoms):
        mesh = g.meshname[len(prefix):] if g.meshname.startswith(prefix) else g.meshname
        if mesh not in _UNDERSIDE_HULLS:
            continue
        if g.contype == 0 and g.conaffinity == 0:          # visual
            if mesh in _JAW_MESHES:
                jaw.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=g.meshname, material=g.material,
                             pos=np.array(g.pos) - MOUTH_PIVOT, quat=g.quat, contype=0, conaffinity=0, group=2)
                spec.delete(g)
        else:                                               # collision hull: replaced by the boxes below
            spec.delete(g)
    # rear underside: forward -0.0385..0.042, up -0.0169..0.0032
    head.add_geom(name=prefix + "head_rear_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=_uf(-0.00685, 0.00175), size=[0.01005, 0.0459, 0.04025], group=3)
    # upper beak (upper pad + front rim of the shell): forward 0.042..0.078, up -0.0092..-0.0013
    head.add_geom(name=prefix + "upper_beak_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=_uf(-0.00525, 0.060), size=[0.00395, 0.0459, 0.018], group=3)
    # jaw plate + lower pad: forward 0.042..0.0777 (the plate centre is cut out behind 0.042 for the neck),
    # up -0.0174..-0.0094 (0.2 mm below the upper box when closed)
    jaw.add_geom(name=prefix + "jaw_plate_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=_uf(-0.0134, 0.05985) - MOUTH_PIVOT, size=[0.0040, 0.0440, 0.01785], group=3)
    # the jaw sits right under the head-yaw bearing and the neck bracket (parent / grandparent of the head)
    for other in ("yaw_roll_motion", "neck_pitch"):
        ex = spec.add_exclude()
        ex.bodyname1 = prefix + "jaw"
        ex.bodyname2 = prefix + other
    head.add_site(name=prefix + "lip_up", pos=_uf(-0.0092, 0.0770), group=4)
    head.add_site(name=prefix + "pad_up", pos=_uf(-0.0092, 0.0680), group=4)
    jaw.add_site(name=prefix + "lip_low", pos=_uf(-0.0134, 0.0777) - MOUTH_PIVOT, group=4)
    jaw.add_site(name=prefix + "jaw_tip_bottom", pos=_uf(-0.0174, 0.0777) - MOUTH_PIVOT, group=4)
    jaw.add_site(name=prefix + "pad_low", pos=_uf(-0.0094, 0.0680) - MOUTH_PIVOT, group=4)

def get_grasp_object_spec(height_m=0.016) -> mujoco.MjSpec:
    if not np.isfinite(height_m) or height_m <= 0:
        raise ValueError("Object height must be positive and finite")
    half = (OBJECT_HALF[0], OBJECT_HALF[1], height_m / 2)
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="object", pos=[0.15, 0.0, half[2]])
    body.add_freejoint(name="object_free")
    body.add_geom(name="object_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half),
                  mass=OBJECT_MASS, rgba=[0.15, 0.45, 0.95, 1.0], condim=4, friction=[1.4, 0.02, 0.0001])
    return spec
