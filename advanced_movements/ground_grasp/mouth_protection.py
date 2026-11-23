"""Mouth closing-torque protection, evaluated on every physics substep.

Torque sign is physical direction (negative closes); angular targets remain 5..35deg.
The threshold is checked on both measured previous-substep motor torque and the
current unsaturated PD demand, so a spike is not applied before detection.
"""
from dataclasses import dataclass
import math
from geometry import MOUTH_MIN_RAD, MOUTH_MAX_RAD, safe_mouth_target


TRIP_RATIO = 1.8
HOLD_RATIO = 1.5


@dataclass
class MouthTorqueGuard:
    baseline_nm: float | None = None
    armed: bool = False
    latched: bool = False
    frozen_target_rad: float | None = None
    trigger_time_s: float | None = None
    absolute_limit_nm: float = .5

    def set_baseline(self, baseline_nm):
        if not math.isfinite(baseline_nm) or baseline_nm <= 0:
            raise ValueError('Closing torque baseline a must be positive and finite')
        if TRIP_RATIO*baseline_nm > self.absolute_limit_nm:
            raise ValueError('1.5a exceeds the existing absolute 0.5 Nm simulation cap')
        self.baseline_nm = float(baseline_nm)

    def reset(self):
        self.armed=False; self.latched=False; self.frozen_target_rad=None; self.trigger_time_s=None

    def update(self, target_rad, angle_rad, velocity_rad_s, measured_nm, time_s):
        if not all(math.isfinite(v) for v in (angle_rad,velocity_rad_s,measured_nm,time_s)):
            raise ValueError('Nonfinite mouth feedback; stop simulation')
        target=safe_mouth_target(target_rad)
        demand=(target-angle_rad)-.01*velocity_rad_s
        closing_demand=max(0.,-demand)
        closing_measured=max(0.,-measured_nm)
        threshold=TRIP_RATIO*self.baseline_nm if self.baseline_nm is not None else self.absolute_limit_nm
        if self.armed and self.baseline_nm is None:
            raise ValueError('Cannot arm without calibrated baseline')
        if self.armed and not self.latched and max(closing_demand,closing_measured)>=threshold:
            self.latched=True
            self.frozen_target_rad=safe_mouth_target(angle_rad)
            self.trigger_time_s=time_s
        if self.latched:
            target=self.frozen_target_rad
            torque=-HOLD_RATIO*self.baseline_nm
        else:
            limit=min(threshold,self.absolute_limit_nm) if self.baseline_nm is not None else self.absolute_limit_nm
            torque=max(-limit,min(limit,demand))
        # Prevent active torque pushing through either angular safety boundary.
        if angle_rad <= MOUTH_MIN_RAD and torque < 0: torque=0.
        if angle_rad >= MOUTH_MAX_RAD and torque > 0: torque=0.
        return dict(target_rad=target,demand_nm=demand,measured_nm=measured_nm,output_nm=torque,
                    threshold_nm=threshold,latched=self.latched,time_s=time_s)
