#############################################################################################
# DON'T CHANGE ANY OF THE BELOW. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################

import os, sys, time, datetime, traceback
import spaceteams as st
def custom_exception_handler(exctype, value, tb):
    error_message = "".join(traceback.format_exception(exctype, value, tb))
    st.logger_fatal(error_message)
sys.excepthook = custom_exception_handler
st.connect_to_sim(sys.argv)

import numpy as np
import numpy.typing as npt


this = st.GetThisSystem()

doSpawning = st.SimGlobals.GetSimEntity().GetParam(st.VarType.bool, "DoSpawning")
if doSpawning:
    while True:
        time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

phobos: st.Entity = this.GetParam(st.VarType.entityRef, "Phobos")
phobosFrame = phobos.GetBodyFixedFrame()
shuttle: st.Entity = this.GetParam(st.VarType.entityRef, "Shuttle")
shuttleFrame = shuttle.GetBodyFixedFrame()
target: st.Entity = this.GetParam(st.VarType.entityRef, "Target")

topCamera: st.Entity = this.GetParam(st.VarType.entityRef, "TopCamera")
topCameraFrame = topCamera.GetBodyFixedFrame()
bottomCamera: st.Entity = this.GetParam(st.VarType.entityRef, "BottomCamera")
bottomCameraFrame = bottomCamera.GetBodyFixedFrame()


def GetNearestDebrisLocations(N: int) -> list[npt.NDArray]:
    allDebrisInfo: list[npt.NDArray] = shuttle.GetParamArray(st.VarType.doubleM3x3, 
                                                            ["Radar", "SortedDebrisData"])
    nearest = allDebrisInfo[:min(N, len(allDebrisInfo))]
    return nearest


def PanTiltCamera(camera: st.Entity, yaw_deg: float, pitch_deg: float) -> None:
    if camera == topCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, -50.0, 0.0)
    elif camera == bottomCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, 0.0, 50.0)
    camera.setRotation_DCM(st.math.Euler321_to_DCM([np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), 0.0]), shuttleFrame)


def SetCommand(accelCommand: npt.NDArray, angAccelCommand: npt.NDArray) -> None:
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAccel"], accelCommand)
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAngAccel"], angAccelCommand)


#############################################################################################
# DON'T CHANGE ANY OF THE ABOVE. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################

# ==============================================================================
# STU4 HIGH-PERFORMANCE AUTOPILOT — RADAR + OPTICAL FUSION DEBRIS AVOIDANCE
#
# Architecture:
#  1. BOOST to CRUISE_SPEED (13 m/s) using full 0.5 m/s^2 thrust
#  2. Active radar debris avoidance — APF repulsion, 360° coverage
#  3. Post-dodge course correction — velocity steering back to target bearing
#  4. OpenCV optical contour scan for large visual threats
#  5. Full LiveDashboard-compatible telemetry output
# ==============================================================================

import json
import cv2

# --- Mission Parameters ---
CRUISE_SPEED:          float = 13.0    # m/s target cruise speed
BOOST_FUEL_LIMIT:      float = 55.0    # kg of fuel allowed for initial boost
NOMINAL_FUEL_KG:       float = 75.0    # for scoring normalisation display
MAX_ACCEL_CMD:         float = 0.48    # m/s^2 (leave slight margin below 0.5 limit)
MAX_ANG_ACCEL:         float = 0.045   # rad/s^2
KEEP_OUT_RADIUS:       float = 50.0    # m (win condition)

# --- Debris Avoidance ---
RADAR_N_CHECK:         int   = 32      # top N nearest radar contacts to check
PROXIMITY_BUBBLE_M:    float = 22.0    # m — engage repulsion if rock surface within this
SAFE_CLEARANCE_M:      float = 6.0     # m — extra safety buffer added to radar diameter
CLOSING_THRESHOLD:     float = -0.05   # m/s — negative = approaching
REPULSION_GAIN:        float = 0.40    # repulsion acceleration scalar
COURSE_CORRECTION_KP:  float = 0.012   # P-gain for velocity-space course-correction
COURSE_CORRECTION_KD:  float = 0.60    # D-gain (damps oscillation)

# --- Optical (OpenCV) ---
CAM_CAPTURE_INTERVAL:  int   = 8       # ticks between optical scans (60Hz → ~7Hz)
OPT_EV:                float = 22.0
OPT_MIN_AREA_PX:       float = 8.0     # filter noise
OPT_OPTICAL_THREAT_AREA_PX: float = 200.0  # large contour = active threat
OPT_CENTER_THREAT_PX:  float = 35.0    # centred contour (pixels from centre)

# --- Attitude gains ---
kp_point: float = 0.25
kd_point: float = 1.00
kd_roll:  float = 1.50

# --- Telemetry ---
TELEM_INTERVAL: int = 6   # ticks between telemetry writes

# ==============================================================================
# Initialisation
# ==============================================================================
PanTiltCamera(topCamera, 0.0, 0.0)
PanTiltCamera(bottomCamera, 0.0, 0.0)

t0_sim = st.SimGlobals.GetSimEntity().GetParam(st.VarType.timestamp, ["Time", "StartDateTAI"])
t0: datetime.datetime = t0_sim.as_datetime()
loop_freq_hz: float = this.GetParam(st.VarType.double, "LoopFreqHz")
tick_dt: float = 1.0 / loop_freq_hz
log_interval: int = max(1, int(loop_freq_hz))

# Initial positions (Phobos frame)
target_state_init = target.getLocVelAcc()
try:
    target_loc = target_state_init.locvel().loc_WRT_ExprIn(phobosFrame)
except Exception:
    target_loc = target_state_init.loc_WRT_ExprIn(phobosFrame)

shuttle_state_init = shuttle.getLocVelAcc()
try:
    initial_shuttle_loc = shuttle_state_init.locvel().loc_WRT_ExprIn(phobosFrame)
except Exception:
    initial_shuttle_loc = shuttle_state_init.loc_WRT_ExprIn(phobosFrame)

initial_dist: float = float(np.linalg.norm(target_loc - initial_shuttle_loc))

telemetry_path = r"d:\Escape_the_Probe\telemetry.json"
telemetry_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetry.json")

# State variables
boost_completed: bool = False
win_registered:  bool = False
iteration:       int   = 0
exit_flag:       bool  = False

# Post-dodge recovery
dodge_active_prev: bool  = False
recovery_counter:  int   = 0
RECOVERY_TICKS:    int   = int(loop_freq_hz * 2.5)  # 2.5s correction window after dodge

# Optical state
optical_contours_cache: list = []
cam_tick: int = 0

# ==============================================================================
# Helper: OpenCV debris scan on front-facing camera
# ==============================================================================
def run_optical_scan() -> list:
    """Return list of [cx_norm, cy_norm, area_px, equiv_r_px] for each contour."""
    results = []
    try:
        PanTiltCamera(topCamera, 0.0, -10.0)   # slight downward look
        time.sleep(0.002)
        frame = topCamera.GetParam(st.VarType.image, ["Camera", "Image"])
        if frame is None:
            return results
        img = np.array(frame, dtype=np.uint8)
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        h, w = gray.shape
        # Otsu threshold (auto handles both lit and dark frames)
        otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu_thresh > 80 or otsu_thresh < 2:
            # near-uniform frame → use fixed threshold
            _, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(gray, int(otsu_thresh), 255, cv2.THRESH_BINARY)
        # Morphological close: fills texture cracks in rocky surfaces
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx_centre = w / 2.0
        cy_centre = h / 2.0
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < OPT_MIN_AREA_PX:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx_px = M["m10"] / M["m00"]
                cy_px = M["m01"] / M["m00"]
            else:
                x_r, y_r, wr, hr = cv2.boundingRect(cnt)
                cx_px = x_r + wr / 2.0
                cy_px = y_r + hr / 2.0
            equiv_r = float(np.sqrt(area / np.pi))
            results.append([cx_px - cx_centre, cy_px - cy_centre, area, equiv_r])
    except Exception:
        pass
    return results


# ==============================================================================
# Main Control Loop
# ==============================================================================
while not exit_flag:
    time.sleep(tick_dt)

    t_now: datetime.datetime = st.SimGlobals.SimClock.GetTimeNow().as_datetime()
    t_from_start: float = (t_now - t0).total_seconds()

    # ---- Vehicle State -------------------------------------------------------
    locVelAcc = shuttle.getLocVelAcc()
    try:
        loc = locVelAcc.locvel().loc_WRT_ExprIn(phobosFrame)
        vel = locVelAcc.locvel().vel_WRT_ExprIn(phobosFrame)
    except Exception:
        loc = locVelAcc.loc_WRT_ExprIn(phobosFrame)
        vel = locVelAcc.vel_WRT_ExprIn(phobosFrame)

    current_AngVel = shuttle.getAngVelocity().WRT(phobosFrame).ExprIn(shuttleFrame)
    current_speed  = float(np.linalg.norm(vel))
    ang_speed      = float(np.linalg.norm(current_AngVel))

    # ---- Fuel & Damage -------------------------------------------------------
    fuel_used:     float = 0.0
    damage_taken:  float = 0.0
    try:
        fuel_used    = float(shuttle.GetParam(st.VarType.double, ["Stats", "FuelUsed"]))
        damage_taken = float(shuttle.GetParam(st.VarType.double, ["Stats", "Damage"]))
    except Exception:
        pass

    # ---- Geometry ------------------------------------------------------------
    to_target_vec  = target_loc - loc
    dist_to_target = float(np.linalg.norm(to_target_vec))
    unit_to_target = to_target_vec / max(dist_to_target, 1e-6)

    # ---- Body-frame attitude error -------------------------------------------
    tgt_in_shuttle = st.frames.FramedVector(to_target_vec, phobosFrame).ExprIn(shuttleFrame)
    tgt_norm       = float(np.linalg.norm(tgt_in_shuttle))
    u_body         = tgt_in_shuttle / max(tgt_norm, 1e-6)
    cos_theta      = np.clip(float(u_body[0]), -1.0, 1.0)
    angle_err_deg  = np.rad2deg(np.arccos(cos_theta))

    # ---- Predicted miss distance (only valid when moving) --------------------
    pred_miss_dist: float = 0.0
    if current_speed >= 2.0:
        vel_unit = vel / current_speed
        # project target relative position onto velocity-perpendicular plane
        d_along = float(np.dot(to_target_vec, vel_unit))
        closest_approach = to_target_vec - d_along * vel_unit
        pred_miss_dist = float(np.linalg.norm(closest_approach))

    # ==========================================================================
    # RADAR DEBRIS AVOIDANCE
    # ==========================================================================
    repulsive_accel  = np.zeros(3)   # Phobos frame
    n_threats        = 0
    nearest_hazard_m = 999.0
    radar_contacts_telem = []

    try:
        nearest_debris = GetNearestDebrisLocations(RADAR_N_CHECK)
        for d in nearest_debris:
            r_vec        = d[0, :]                     # relative position (Phobos frame)
            r_dist       = float(np.linalg.norm(r_vec))
            d_closest    = float(d[1, 0])              # range to rock surface (m)
            closing_rate = float(d[1, 1])              # negative = approaching
            rock_diam    = float(d[1, 2])              # radar diameter (m)
            rock_radius  = rock_diam / 2.0

            # Safety margin: account for noisy diameter + buffer
            effective_clearance = d_closest - SAFE_CLEARANCE_M

            if d_closest < nearest_hazard_m:
                nearest_hazard_m = d_closest

            # Treat as threat if closing or slow (slow = can't track closing rate well)
            is_closing = (closing_rate < CLOSING_THRESHOLD) or (current_speed < 2.0)

            if effective_clearance < PROXIMITY_BUBBLE_M and is_closing:
                n_threats += 1
                # Repulsion direction: away from rock
                n_away = -r_vec / max(r_dist, 1e-4)
                # Remove component along target direction (preserve forward progress)
                n_lateral = n_away - np.dot(n_away, unit_to_target) * unit_to_target
                lat_norm = float(np.linalg.norm(n_lateral))
                if lat_norm > 1e-4:
                    n_lateral = n_lateral / lat_norm
                else:
                    n_lateral = n_away   # rock exactly on course — dodge sideways fully

                # Strength scales with proximity: 1.0 at surface, 0 at bubble edge
                penetration = max(0.0, PROXIMITY_BUBBLE_M - effective_clearance) / PROXIMITY_BUBBLE_M
                strength = REPULSION_GAIN * penetration * penetration
                repulsive_accel += strength * n_lateral

            # Build telemetry radar contacts list (body frame offset for dashboard)
            r_body = st.frames.FramedVector(r_vec, phobosFrame).ExprIn(shuttleFrame)
            radar_contacts_telem.append([float(r_body[0]), float(r_body[1]), rock_diam, closing_rate])
    except Exception:
        pass

    dodge_active = float(np.linalg.norm(repulsive_accel)) > 0.01

    # Clamp repulsion magnitude
    rep_mag = float(np.linalg.norm(repulsive_accel))
    if rep_mag > MAX_ACCEL_CMD * 0.85:
        repulsive_accel = (MAX_ACCEL_CMD * 0.85 / rep_mag) * repulsive_accel

    # ==========================================================================
    # OPTICAL CONTOUR SCAN (every CAM_CAPTURE_INTERVAL ticks)
    # ==========================================================================
    cam_tick += 1
    if cam_tick >= CAM_CAPTURE_INTERVAL:
        cam_tick = 0
        optical_contours_cache = run_optical_scan()

    # Optical threat: large and centred contour → adds lateral push
    optical_threat_accel = np.zeros(3)
    opt_threat_found = False
    forward_obs_telem = []
    for cnt_data in optical_contours_cache:
        cx_off, cy_off, area, equiv_r = cnt_data
        offset_px = float(np.hypot(cx_off, cy_off))
        if area > OPT_OPTICAL_THREAT_AREA_PX and offset_px < OPT_CENTER_THREAT_PX:
            # Rock roughly ahead and large — push slightly perpendicular
            # cx_off > 0 means rock is right of centre → push left (body -Y)
            push_body = np.array([0.0, -np.sign(cx_off) if abs(cx_off) > 2.0 else -1.0,
                                  -np.sign(cy_off) if abs(cy_off) > 2.0 else 0.0])
            push_phobos = np.array(st.frames.FramedVector(push_body, shuttleFrame).ExprIn(phobosFrame))
            optical_threat_accel += 0.08 * push_phobos
            opt_threat_found = True
        # Build forward observation list for dashboard
        forward_obs_telem.append([cx_off, cy_off, 0.0, equiv_r])

    # Fused matches list (for dashboard)
    fused_matches_telem = []

    # ==========================================================================
    # TRANSLATIONAL GUIDANCE
    # ==========================================================================
    # Phase logic
    if dist_to_target <= KEEP_OUT_RADIUS:
        current_phase = "WON"
    elif not boost_completed:
        if current_speed >= CRUISE_SPEED or fuel_used >= BOOST_FUEL_LIMIT:
            boost_completed = True
            current_phase = "COAST"
        else:
            current_phase = "BOOST"
    else:
        if dodge_active or opt_threat_found:
            current_phase = "EVADE"
        elif recovery_counter > 0:
            current_phase = "RECOVER"
        else:
            current_phase = "COAST"

    # Track dodge transitions for recovery window
    if dodge_active and not dodge_active_prev:
        recovery_counter = RECOVERY_TICKS   # start recovery timer when dodge begins
    if not dodge_active and dodge_active_prev and recovery_counter <= 0:
        recovery_counter = RECOVERY_TICKS   # also reset when dodge clears
    if recovery_counter > 0:
        recovery_counter -= 1
    dodge_active_prev = dodge_active

    # Compute linear acceleration command (Phobos frame)
    linearAccel_phobos = np.zeros(3)

    if current_phase == "WON":
        linearAccel_phobos = np.zeros(3)

    elif current_phase == "BOOST":
        # Velocity error toward target at cruise speed
        goal_vel = CRUISE_SPEED * unit_to_target
        vel_err  = goal_vel - vel
        a_cmd    = 1.0 * vel_err  # P-controller on velocity
        # During early alignment, limit to 0.15 m/s^2 to not overshoot sideways
        max_a = MAX_ACCEL_CMD if angle_err_deg < 20.0 else 0.15
        a_mag = float(np.linalg.norm(a_cmd))
        if a_mag > max_a:
            a_cmd = (max_a / a_mag) * a_cmd
        linearAccel_phobos = a_cmd + repulsive_accel + optical_threat_accel

    elif current_phase in ("EVADE", "COAST-EVADE"):
        # Apply repulsion only — no throttle (preserves fuel during coast)
        linearAccel_phobos = repulsive_accel + optical_threat_accel

    elif current_phase == "RECOVER":
        # Steer velocity back toward target bearing
        if current_speed >= 1.0:
            current_vel_unit = vel / current_speed
            bearing_err = unit_to_target - current_vel_unit
            # Only correct if bearing error is significant (> ~3°)
            bearing_err_mag = float(np.linalg.norm(bearing_err))
            if bearing_err_mag > 0.05:
                correct_dir = bearing_err / bearing_err_mag
                # Soft correction proportional to bearing error, damp with angular velocity
                correction_a = COURSE_CORRECTION_KP * current_speed * bearing_err_mag * correct_dir
                corr_mag = float(np.linalg.norm(correction_a))
                if corr_mag > 0.12:
                    correction_a = (0.12 / corr_mag) * correction_a
                linearAccel_phobos = correction_a
        # Still push away from any rocks even in recovery
        if dodge_active:
            linearAccel_phobos += repulsive_accel

    else:  # COAST — pure ballistic, save all fuel
        linearAccel_phobos = np.zeros(3)
        # Still dodge if something pops up
        if dodge_active or opt_threat_found:
            linearAccel_phobos = repulsive_accel + optical_threat_accel
            current_phase = "COAST-EVADE"

    # Clamp total linear accel
    total_a_mag = float(np.linalg.norm(linearAccel_phobos))
    if total_a_mag > MAX_ACCEL_CMD:
        linearAccel_phobos = (MAX_ACCEL_CMD / total_a_mag) * linearAccel_phobos

    # Express in shuttle body frame for the actuator
    linearAcceleration = st.frames.FramedVector(linearAccel_phobos, phobosFrame).ExprIn(shuttleFrame)

    # ==========================================================================
    # ATTITUDE CONTROL — 2-DOF pointing + roll damp
    # ==========================================================================
    # Deadband: relax thrusters when coasting and well aligned
    attitude_deadband = (
        current_phase in ("COAST",) and
        angle_err_deg < 1.5 and
        ang_speed < 0.008
    )

    if attitude_deadband:
        angularAcceleration = np.zeros(3)
    else:
        alpha_x = -kd_roll * float(current_AngVel[0])
        alpha_y = -kp_point * float(u_body[2]) - kd_point * float(current_AngVel[1])
        alpha_z =  kp_point * float(u_body[1]) - kd_point * float(current_AngVel[2])
        ang_acc = np.array([alpha_x, alpha_y, alpha_z])
        ang_mag = float(np.linalg.norm(ang_acc))
        if ang_mag > MAX_ANG_ACCEL:
            ang_acc = (MAX_ANG_ACCEL / ang_mag) * ang_acc
        angularAcceleration = ang_acc

    # Send command
    SetCommand(linearAcceleration, angularAcceleration)

    # ==========================================================================
    # CAMERA GIMBAL — track cycler
    # ==========================================================================
    cam_body = st.frames.FramedVector(to_target_vec, phobosFrame).ExprIn(shuttleFrame)
    cg_x = float(cam_body[0])
    cg_y = float(cam_body[1])
    cg_z = float(cam_body[2])
    if cg_x > 1.0:
        cam_yaw   = float(np.clip(np.rad2deg(np.arctan2(cg_y, cg_x)), -85.0, 85.0))
        cam_pitch = float(np.clip(np.rad2deg(np.arctan2(-cg_z, np.hypot(cg_x, cg_y))), -50.0, 0.0))
        PanTiltCamera(topCamera, cam_yaw, cam_pitch)
        PanTiltCamera(bottomCamera, cam_yaw, 0.0)

    # ==========================================================================
    # TELEMETRY — full LiveDashboard compatible output
    # ==========================================================================
    if iteration % log_interval == 0:
        msg = (
            f"[{current_phase}] T={t_from_start:.1f}s|Dist={dist_to_target:.1f}m|"
            f"Miss={pred_miss_dist:.1f}m|Spd={current_speed:.2f}m/s|"
            f"Threats={n_threats}|Dmg={damage_taken:.0f}J|"
            f"Fuel={fuel_used:.3f}kg/{NOMINAL_FUEL_KG}kg"
        )
        st.OnScreenLogMessage(msg, "Autopilot", st.Severity.Info)

    if iteration % TELEM_INTERVAL == 0:
        rep_body_list = [float(repulsive_accel[0]), float(repulsive_accel[1]), float(repulsive_accel[2])]
        telem = {
            "time":             t_from_start,
            "dist":             dist_to_target,
            "pred_miss":        pred_miss_dist,
            "speed":            current_speed,
            "fuel":             fuel_used,
            "dmg":              damage_taken,
            "phase":            current_phase,
            "threats":          n_threats,
            "won":              win_registered,
            "shuttle_x":        float(loc[0]),
            "shuttle_z":        float(loc[2]),
            "target_x":         float(target_loc[0]),
            "target_z":         float(target_loc[2]),
            "rep_body":         rep_body_list,
            "radar_contacts":   radar_contacts_telem[:20],
            "forward_obs":      forward_obs_telem[:10],
            "optical_contours": [[c[0], c[1], c[2], c[3]] for c in optical_contours_cache[:10]],
            "fused_matches":    fused_matches_telem,
        }
        try:
            with open(telemetry_path, "w", encoding="utf-8") as tf:
                json.dump(telem, tf)
        except Exception:
            pass
        try:
            with open(telemetry_path2, "w", encoding="utf-8") as tf:
                json.dump(telem, tf)
        except Exception:
            pass

    
    if dist_to_target <= KEEP_OUT_RADIUS:
        if not win_registered:
            win_registered = True
            st.OnScreenAlert(
                f"[VICTORY] Dist={dist_to_target:.2f}m | T={t_from_start:.2f}s | "
                f"Fuel={fuel_used:.2f}kg | Dmg={damage_taken:.0f}J",
                "Victory",
                st.Severity.Warning
            )
        SetCommand(np.zeros(3), np.zeros(3))
        time.sleep(0.3)
        exit_flag = True

    iteration += 1

st.leave_sim()
