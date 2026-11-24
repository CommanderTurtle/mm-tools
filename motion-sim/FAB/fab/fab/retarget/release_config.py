"""The released G1 retargeter configuration, used by ``fab-retarget``.

SOLVE: the scene objective's settings for bench/videomimic/scene_retarget.py (``--cfg``): foot / heel / toe contact groups,
acceleration weight 120, no foot-height or swing terms, self-collision on, the root initialised on the terrain, the human's up
axis from the template pose, 1600 solver iterations. The contact labels come from the mesh rule (a foot part's lowest vertex
within H_CONTACT of the terrain and slower than V_CONTACT).
SIGMA_IN: the solver's input smoother (``--sigma_in``, frames at 50 fps) on every input.
PIN: the stance pin's settings for bench/stance_pin.py (``--cfg``): L2 smoothing + sole footprint, a 4.5 cm / frame root-step cap
(w_jump), drift cap 8 cm, turn cap 15 deg.
Modes (fab-retarget --mode): mesh is SOLVE; points (the default) is SOLVE plus the dense-point term (W_PTS;
fab.retarget.dense_corr's correspondence, each group's bone turned onto the skeleton's), with contacts, stance pin and
everything else as mesh; skeleton is SOLVE with the mesh rule's three uses (contacts, seated pelvis, root start) replaced by
joint rules (fab/retarget/skeleton.py) at the same H_CONTACT / V_CONTACT.
tests/data/retarget_config.json holds a frozen copy (tests/test_retarget_config.py).
"""

SOLVE = {"lr_t": 0.002, "w_foot_h": 0.0, "w_acc": 120.0, "w_self": 1000.0, "self_legs": 1, "w_swing": 0.0, "swing_clear": 0.03,
         "flex_init": 1, "root_lowest": 0, "world_support": "contact", "terr_grad_sigma_m": 0.033,
         "contact_groups": "lfoot,rfoot,lheel,ltoe,rheel,rtoe", "root_init": "terrain", "iters": 1600, "hum_up": "template"}
H_CONTACT = 0.04  # m
V_CONTACT = 0.15  # m/s
SIGMA_IN = 8.0
PIN = {"w_smooth": 30000, "lr_root": 0.2, "drift_cap": 0.08, "lift": 0.01, "w_jump": 10000000, "rot_cap_deg": 15, "retry_mm": 3,
       "foot_k": 20, "cap_eps": 0.001}


def solve_cfg():
    """The solver config as fab-retarget passes it (the mesh rule's thresholds on the scene terrain included)."""
    return dict(SOLVE, h_contact=H_CONTACT, v_contact=V_CONTACT)


W_PTS = 1000.0  # the dense-point term's weight (points mode)


def solve_cfg_dp():
    """The points mode's solver config: solve_cfg() plus the dense-point term."""
    return dict(solve_cfg(), w_pts=float(W_PTS))


# fab-retarget-ee (scripts/fab_retarget_ee.py; docs/retarget_ee.md; flat floor only): the same solver and stance pin with the
# end-effector position terms added. EE_W: the hand / foot position weights (squared distance, m^2; as W_PTS); the
# bone-direction and dense-point terms x EE_SECONDARY (1: as fab-retarget's). EE_HAND_H: the hand-floor windows' height rule (the
# lowest hand vertex within this of the floor), debounced as the stance pin's runs (EE_GAP_S filled, EE_RUN_S dropped).
EE_W = {"w_ee_hand": 1000.0, "w_ee_foot": 100.0}  # feet second: stance is the contact terms + pin
EE_SECONDARY = 1.0
EE_HAND_H = 0.04  # m
EE_GAP_S, EE_RUN_S = 0.1, 0.2


def solve_cfg_ee(mode="points", secondary=None):
    """fab-retarget-ee's solver config for a mode: that mode's fab-retarget config, the end-effector weights added, the
    direction (w_dir) and dense-point (w_pts) weights x EE_SECONDARY (or the given factor)."""
    from bench.videomimic.scene_retarget import CORE

    f = EE_SECONDARY if secondary is None else float(secondary)
    base = solve_cfg_dp() if mode == "points" else solve_cfg()
    cfg = dict(base, **EE_W, w_dir=float(CORE["w_dir"]) * f)
    if "w_pts" in cfg:
        cfg["w_pts"] = float(cfg["w_pts"]) * f
    return cfg


# --ground_aware (fab-retarget / fab-retarget-ee, off by default; docs/retarget.md). Contact labels from fab-world's video contact
# model (<work>/contacts): a foot is planted when its heel or toe static-contact probability is >= GA_THR, debounced as the stance
# pin's runs; a planted foot is held at the terrain height under it and does not slide (the contact / skate terms). No flight: on
# frames with no foot planted where the model says a foot touches (any-contact >= GA_THR) or the person's lowest foot vertex is
# < GA_NO_FLIGHT above the terrain, the lower G1 foot is held at the terrain. With end-effector targets (fab-retarget-ee) the contact
# weight is GA_W_CONTACT_EE x the largest target weight, so no target lifts a planted foot (else the released w_contact). The stance
# pin holds the same labelled feet within its lift of the heightfield under them (GA_W_GROUND), so it does not lift them again.
GA_THR = 0.5
GA_NO_FLIGHT = 0.10  # m
GA_W_CONTACT_EE = 100.0
GA_W_GROUND = 1e5


def ground_aware_cfg(cfg, contacts_dir):
    """cfg with --ground_aware's settings; contacts_dir: fab-world's work/contacts (<clip>__0.contact.npz)."""
    out = dict(cfg, contacts_dir=str(contacts_dir), contact_key="video_static_prob", contact_mode="replace", contact_thr=GA_THR,
               contact_debounce=1, no_flight=GA_NO_FLIGHT, no_flight_key="video_any_prob", contact_groups=cfg["contact_groups"] + ",anyfoot")
    wee = max(float(cfg.get(k) or 0.0) for k in ("w_ee_hand", "w_ee_foot", "w_hand_t"))
    return dict(out, w_contact=GA_W_CONTACT_EE * wee) if wee > 0 else out


def ground_aware_pin(pin):
    """The stance pin's settings with --ground_aware's hold."""
    return dict(pin, w_ground=GA_W_GROUND)
