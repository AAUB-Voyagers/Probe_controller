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
# SPACE TEAMS STU4 - HIGH-PRECISION AUTONOMOUS GUIDANCE & SENSOR FUSION APF
#
# Key Features:
# 1. Pad Liftoff & Attitude Priority:
#    - 0.15 m/s^2 gentle liftoff while pitching up, accelerating to 0.48 m/s^2
#      once aligned forward (< 18°), cutting off cleanly at CRUISE_SPEED.
# 2. 2-DOF Direct LOS Attitude Control (Zero Roll Spin):
#    - Pitch: -kp * u_body[2] - kd * w_y
#    - Yaw:   +kp * u_body[1] - kd * w_z
#    - Roll:  -kd_roll * w_x (zero spin)
# 3. Enhanced Lateral APF + Radar-Optical Fusion:
#    - Null-space lateral repulsion for collision avoidance without speed loss.
#    - OpenCV optical contour extraction & association with radar tracks.
# 4. Detached LiveDashboard HUD:
#    - Streams telemetry to real-time HUD with 0.0 ms impact on 60 Hz sim loop.
# ==============================================================================

import json
import cv2

# --- Mission & Guidance Parameters ---
CRUISE_SPEED:          float = 13.00   # m/s target cruise speed
BOOST_FUEL_LIMIT:      float = 55.00   # kg fuel limit for initial boost phase
NOMINAL_FUEL_KG:       float = 75.00   # kg reference fuel for scoring display
MAX_ACCEL_CMD:         float = 0.48    # m/s^2 linear acceleration limit
MAX_ANG_ACCEL:         float = 0.045   # rad/s^2 angular acceleration limit
KEEP_OUT_RADIUS:       float = 50.0    # m win condition radius

# --- Debris Avoidance ---
RADAR_N_CHECK:         int   = 100      # nearest radar contacts to inspect
PROXIMITY_BUBBLE_M:    float = 22.0    # m proximity threshold for active repulsion
SAFE_CLEARANCE_M:      float = 6.0     # m buffer added to debris radius
CLOSING_THRESHOLD:     float = -0.05   # m/s closing speed threshold
REPULSION_GAIN:        float = 0.40    # lateral repulsion acceleration scalar
COURSE_CORRECTION_KP:  float = 0.012   # P-gain for post-dodge velocity recovery
COURSE_CORRECTION_KD:  float = 0.60    # D-gain damping

# --- Optical Sensing (OpenCV) ---
CAM_CAPTURE_INTERVAL:  int   = 8       # ticks between optical scans (~7.5 Hz at 60 Hz)
OPT_EV:                float = 10.0    # EV 10 calibrated exposure
OPT_MIN_AREA_PX:       float = 6.0     # min contour area in pixels
OPT_OPTICAL_THREAT_AREA_PX: float = 120.0  # large contour threat threshold
OPT_CENTER_THREAT_PX:  float = 30.0    # distance from optical axis

# --- Attitude Control Gains ---
kp_point: float = 0.25
kd_point: float = 1.00
kd_roll:  float = 1.50

# --- Telemetry & Dashboard Mode ---
TELEM_INTERVAL:        int  = 6        # ticks between telemetry writes (10 Hz)
ENABLE_DASHBOARD:      bool = True     # auto-launch dashboard window on start
ENABLE_LIVE_DASHBOARD: bool = True     # alias for compatibility
DASHBOARD_TYPE:        str  = "RADAR"  # Options: "TRAJECTORY" (or "PATH"), "RADAR", "LIVE"

# ==============================================================================
# Initialization
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


def LaunchDashboardIfIdle() -> None:
    """Automatically spawn selected dashboard as a detached background process if not already open."""
    import socket
    import subprocess

    dtype = str(globals().get("DASHBOARD_TYPE", "RADAR")).upper()
    if dtype in ("TRAJECTORY", "PATH", "LOS"):
        port = 49283
        script_name = "TrajectoryDashboard.py"
    elif dtype == "RADAR":
        port = 49282
        script_name = "RadarDashboard.py"
    else:
        port = 49281
        script_name = "LiveDashboard.py"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.12)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return  # Dashboard window is already open and running
    except Exception:
        pass

    try:
        # script_name determined above
        this_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else r"d:\Escape_the_Probe"
        dashboard_paths = [
            os.path.join(this_dir, script_name),
            os.path.join(r"d:\Escape_the_Probe", script_name),
            os.path.join(r"d:\Escape_the_Probe\Repos\STU4-Solution", script_name),
            os.path.join(r"C:\Users\usser\AppData\Local\SimDynamX\STPro\Assets\Repos\STU4-Solution", script_name)
        ]
        target_script = None
        for p in dashboard_paths:
            if os.path.exists(p):
                target_script = p
                break

        if target_script is not None:
            pyw_exe = sys.executable.replace("python.exe", "pythonw.exe")
            exe = pyw_exe if os.path.exists(pyw_exe) else sys.executable
            DETACHED_FLAGS = 0x00000008 | 0x00000200
            subprocess.Popen([exe, target_script], creationflags=DETACHED_FLAGS, close_fds=True)
    except Exception:
        pass


if globals().get("ENABLE_DASHBOARD", True) or globals().get("ENABLE_LIVE_DASHBOARD", False):
    LaunchDashboardIfIdle()

# State variables
boost_completed:   bool = False
win_registered:    bool = False
iteration:         int  = 0
exit_flag:         bool = False
dodge_active_prev: bool = False
recovery_counter:  int  = 0
RECOVERY_TICKS:    int  = int(loop_freq_hz * 2.5)  # 2.5s post-dodge correction window

# Optical state
optical_contours_cache: list = []
cam_tick: int = 0

cameraParams = st.CaptureImageProperties()
cameraParams.nonphys_EV = OPT_EV
cameraParams.FOV = 60.0
cameraParams.ResolutionX = 128
cameraParams.ResolutionY = 128


def run_optical_scan() -> list:
    """Capture camera frame and return list of [cx, cy, area_px, equiv_r_px] in pixel coordinates."""
    results = []
    try:
        image = st.camera.CaptureImageBlocking(topCamera, cameraParams)
        if image is not None:
            rgb8 = image.as_RGB8()
            im_r = np.asarray(rgb8.PixelsR).reshape((128, 128))
            im_g = np.asarray(rgb8.PixelsG).reshape((128, 128))
            im_b = np.asarray(rgb8.PixelsB).reshape((128, 128))
            gray = np.clip(0.299 * im_r + 0.587 * im_g + 0.114 * im_b, 0, 255).astype(np.uint8)

            otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_val = max(15, min(int(otsu_thresh), 140))
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < OPT_MIN_AREA_PX:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx_px_val = float(M["m10"] / M["m00"])
                    cy_px_val = float(M["m01"] / M["m00"])
                else:
                    xr, yr, wr, hr = cv2.boundingRect(cnt)
                    cx_px_val = float(xr + wr / 2.0)
                    cy_px_val = float(yr + hr / 2.0)
                equiv_r = float(np.sqrt(area / np.pi))
                results.append([cx_px_val, cy_px_val, area, equiv_r])
    except Exception:
        pass
    return results


# ==============================================================================
# Main Simulation Loop
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

    # ---- Geometry & Target Vector --------------------------------------------
    to_target_vec  = target_loc - loc
    dist_to_target = float(np.linalg.norm(to_target_vec))
    unit_to_target = to_target_vec / max(dist_to_target, 1e-6)

    # ---- Body-Frame Line-of-Sight Attitude Error -----------------------------
    tgt_in_shuttle = st.frames.FramedVector(to_target_vec, phobosFrame).ExprIn(shuttleFrame)
    tgt_norm       = float(np.linalg.norm(tgt_in_shuttle))
    u_body         = tgt_in_shuttle / max(tgt_norm, 1e-6)
    cos_theta      = np.clip(float(u_body[0]), -1.0, 1.0)
    angle_err_deg  = np.rad2deg(np.arccos(cos_theta))

    # ---- Predicted Miss Distance ---------------------------------------------
    pred_miss_dist: float = 0.0
    if current_speed >= 2.0:
        vel_unit = vel / current_speed
        d_along = float(np.dot(to_target_vec, vel_unit))
        closest_approach = to_target_vec - d_along * vel_unit
        pred_miss_dist = float(np.linalg.norm(closest_approach))

    # ==========================================================================
    # 1. OPTICAL CONTOUR SCAN (every CAM_CAPTURE_INTERVAL ticks)
    # ==========================================================================
    cam_tick += 1
    if cam_tick >= CAM_CAPTURE_INTERVAL:
        cam_tick = 0
        optical_contours_cache = run_optical_scan()

    # ==========================================================================
    # 2. RADAR & OPTICAL SENSOR FUSION APF
    # ==========================================================================
    repulsive_accel      = np.zeros(3)   # Phobos frame
    n_threats            = 0
    nearest_hazard_m     = 999.0
    radar_contacts_telem = []
    forward_obs_telem    = []
    fused_matches_telem  = []

    CAM_RES = 128
    cx_px = 64.0
    cy_px = 64.0
    focal_px = (CAM_RES * 0.5) / np.tan(np.deg2rad(60.0 * 0.5))  # ~110.85 px

    try:
        nearest_debris = GetNearestDebrisLocations(RADAR_N_CHECK)
        for d in nearest_debris:
            r_vec        = d[0, :]                     # relative position (Phobos frame)
            r_dist       = float(np.linalg.norm(r_vec))
            d_closest    = float(d[1, 0])              # range to rock surface (m)
            closing_rate = float(d[1, 1])              # negative = approaching
            rock_diam    = float(d[1, 2])              # radar diameter (m)
            rock_radius  = max(0.5, rock_diam / 2.0)

            # Express in shuttle body frame
            r_body = st.frames.FramedVector(r_vec, phobosFrame).ExprIn(shuttleFrame)
            x_b, y_b, z_b = float(r_body[0]), float(r_body[1]), float(r_body[2])

            radar_contacts_telem.append([x_b, y_b, rock_diam, closing_rate, z_b, d_closest, r_dist])

            # Forward obstacles in flight path
            if 1.0 < x_b < 150.0:
                forward_obs_telem.append([x_b, y_b, z_b, rock_radius])

            # SENSOR FUSION: Match forward radar obstacle with optical contours
            if x_b > 2.0:
                u_pred = cx_px + focal_px * (y_b / x_b)
                v_pred = cy_px - focal_px * (z_b / x_b)

                if 0.0 <= u_pred <= CAM_RES and 0.0 <= v_pred <= CAM_RES:
                    best_match = None
                    best_dist_px = 28.0
                    for (ocx, ocy, area, eq_r) in optical_contours_cache:
                        d_px = float(np.hypot(ocx - u_pred, ocy - v_pred))
                        if d_px < best_dist_px:
                            best_dist_px = d_px
                            best_match = (ocx, ocy, eq_r)

                    if best_match is not None:
                        opt_cx, opt_cy, opt_r = best_match
                        y_opt = (opt_cx - cx_px) * x_b / focal_px
                        z_opt = -(opt_cy - cy_px) * x_b / focal_px
                        fused_body = np.array([x_b, y_opt, z_opt])
                        fused_phobos = np.array(st.frames.FramedVector(fused_body, shuttleFrame).ExprIn(phobosFrame))
                        fused_matches_telem.append([
                            float(u_pred), float(v_pred), float(opt_cx), float(opt_cy), float(x_b), float(rock_radius)
                        ])
                        # Refine obstacle vector using optical centroid
                        r_vec = fused_phobos
                        r_dist = float(np.linalg.norm(r_vec))

            # Safety margin & threat calculation
            effective_clearance = d_closest - SAFE_CLEARANCE_M
            if d_closest < nearest_hazard_m:
                nearest_hazard_m = d_closest

            is_closing = (closing_rate < CLOSING_THRESHOLD) or (current_speed < 2.0)
            if effective_clearance < PROXIMITY_BUBBLE_M and is_closing:
                n_threats += 1
                n_away = -r_vec / max(r_dist, 1e-4)
                # Null-space lateral projection (preserves forward momentum)
                n_lateral = n_away - np.dot(n_away, unit_to_target) * unit_to_target
                lat_norm = float(np.linalg.norm(n_lateral))
                if lat_norm > 1e-4:
                    n_lateral = n_lateral / lat_norm
                else:
                    n_lateral = n_away

                penetration = max(0.0, PROXIMITY_BUBBLE_M - effective_clearance) / PROXIMITY_BUBBLE_M
                strength = REPULSION_GAIN * (penetration ** 2)
                repulsive_accel += strength * n_lateral
    except Exception:
        pass

    dodge_active = float(np.linalg.norm(repulsive_accel)) > 0.01

    # Clamp repulsion acceleration
    rep_mag = float(np.linalg.norm(repulsive_accel))
    if rep_mag > MAX_ACCEL_CMD * 0.85:
        repulsive_accel = (MAX_ACCEL_CMD * 0.85 / rep_mag) * repulsive_accel

    # Optical direct threat backup
    optical_threat_accel = np.zeros(3)
    opt_threat_found = False
    for (cx_val, cy_val, area, equiv_r) in optical_contours_cache:
        cx_off = cx_val - cx_px
        cy_off = cy_val - cy_px
        offset_px = float(np.hypot(cx_off, cy_off))
        if area > OPT_OPTICAL_THREAT_AREA_PX and offset_px < OPT_CENTER_THREAT_PX:
            push_body = np.array([0.0, -np.sign(cx_off) if abs(cx_off) > 2.0 else -1.0,
                                  -np.sign(cy_off) if abs(cy_off) > 2.0 else 0.0])
            push_phobos = np.array(st.frames.FramedVector(push_body, shuttleFrame).ExprIn(phobosFrame))
            optical_threat_accel += 0.08 * push_phobos
            opt_threat_found = True

    # ==========================================================================
    # 3. TRANSLATIONAL GUIDANCE & THRUSTER COMMAND
    # ==========================================================================
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

    if dodge_active and not dodge_active_prev:
        recovery_counter = RECOVERY_TICKS
    if not dodge_active and dodge_active_prev and recovery_counter <= 0:
        recovery_counter = RECOVERY_TICKS
    if recovery_counter > 0:
        recovery_counter -= 1
    dodge_active_prev = dodge_active

    linearAccel_phobos = np.zeros(3)

    if current_phase == "WON":
        linearAccel_phobos = np.zeros(3)

    elif current_phase == "BOOST":
        goal_vel = CRUISE_SPEED * unit_to_target
        vel_err  = goal_vel - vel
        a_cmd    = 1.0 * vel_err
        # Smooth pad liftoff: 0.15 m/s^2 while pitching up, full thrust once aligned (< 18°)
        max_a = MAX_ACCEL_CMD if angle_err_deg < 18.0 else 0.15
        a_mag = float(np.linalg.norm(a_cmd))
        if a_mag > max_a:
            a_cmd = (max_a / a_mag) * a_cmd
        linearAccel_phobos = a_cmd + repulsive_accel + optical_threat_accel

    elif current_phase in ("EVADE", "COAST-EVADE"):
        # Apply repulsion maneuvers only (zero forward throttle during coast)
        linearAccel_phobos = repulsive_accel + optical_threat_accel

    elif current_phase == "RECOVER":
        if current_speed >= 1.0:
            current_vel_unit = vel / current_speed
            bearing_err = unit_to_target - current_vel_unit
            bearing_err_mag = float(np.linalg.norm(bearing_err))
            if bearing_err_mag > 0.05:
                correct_dir = bearing_err / bearing_err_mag
                correction_a = COURSE_CORRECTION_KP * current_speed * bearing_err_mag * correct_dir
                corr_mag = float(np.linalg.norm(correction_a))
                if corr_mag > 0.12:
                    correction_a = (0.12 / corr_mag) * correction_a
                linearAccel_phobos = correction_a
        if dodge_active:
            linearAccel_phobos += repulsive_accel

    else:  # COAST
        linearAccel_phobos = np.zeros(3)
        if dodge_active or opt_threat_found:
            linearAccel_phobos = repulsive_accel + optical_threat_accel
            current_phase = "COAST-EVADE"

    total_a_mag = float(np.linalg.norm(linearAccel_phobos))
    if total_a_mag > MAX_ACCEL_CMD:
        linearAccel_phobos = (MAX_ACCEL_CMD / total_a_mag) * linearAccel_phobos

    # Transform linear acceleration into shuttle body frame
    linearAcceleration = st.frames.FramedVector(linearAccel_phobos, phobosFrame).ExprIn(shuttleFrame)

    # ==========================================================================
    # 4. ATTITUDE CONTROL — 2-DOF Direct Pointing + Zero Roll Spin
    # ==========================================================================
    if boost_completed and angle_err_deg < 1.0 and ang_speed < 0.005:
        angularAcceleration = np.zeros(3)
    else:
        # Zero roll rate, pitch and yaw directly driven by LOS error vector
        alpha_x = -kd_roll * current_AngVel[0]
        alpha_y = -kp_point * float(u_body[2]) - kd_point * current_AngVel[1]
        alpha_z = kp_point * float(u_body[1]) - kd_point * current_AngVel[2]
        ang_acc = np.array([alpha_x, alpha_y, alpha_z])

        ang_acc_mag = float(np.linalg.norm(ang_acc))
        if ang_acc_mag > MAX_ANG_ACCEL:
            ang_acc = (MAX_ANG_ACCEL / ang_acc_mag) * ang_acc
        angularAcceleration = ang_acc

    # DELIVER COMMAND TO ACTUATOR
    SetCommand(linearAcceleration, angularAcceleration)

    # ==========================================================================
    # 5. TELEMETRY & LOGGING
    # ==========================================================================
    if iteration % log_interval == 0:
        msg = (
            f"[{current_phase}] T={t_from_start:.1f}s | Dist={dist_to_target:.1f}m | "
            f"Miss={pred_miss_dist:.1f}m | Spd={current_speed:.2f}m/s | "
            f"Threats={n_threats} | Dmg={damage_taken:.0f}J | "
            f"Fuel={fuel_used:.2f}kg/{NOMINAL_FUEL_KG}kg"
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
            "radar_contacts":   radar_contacts_telem[:100],
            "forward_obs":      forward_obs_telem[:16],
            "optical_contours": [[c[0], c[1], c[2], c[3]] for c in optical_contours_cache[:16]],
            "fused_matches":    fused_matches_telem[:12],
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

    # ==========================================================================
    # 6. VICTORY CHECK
    # ==========================================================================
    if dist_to_target <= KEEP_OUT_RADIUS:
        if not win_registered:
            win_registered = True
            st.OnScreenAlert(
                f"[VICTORY ACHIEVED] Dist={dist_to_target:.2f}m <= 50m! Arrival Time={t_from_start:.2f}s | "
                f"Fuel Used={fuel_used:.2f}kg | Damage={damage_taken:.0f}J",
                "Victory",
                st.Severity.Warning
            )
        SetCommand(np.zeros(3), np.zeros(3))
        time.sleep(0.3)
        exit_flag = True

    iteration += 1

st.leave_sim()
