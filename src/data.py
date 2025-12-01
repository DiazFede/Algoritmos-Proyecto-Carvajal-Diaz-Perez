from dataclasses import dataclass
import numpy as np

@dataclass
class Satellite:
    sid: int
    e_max: float      # energy budget per planning horizon
    t_max: float      # max operable time
    slew_coeff: float # unused in este modelo (0)

@dataclass
class Target:
    tid: int
    priority: float    # w_t
    critical: bool
    min_obs: int
    x: float
    y: float

@dataclass
class Opportunity:
    oid: int
    sat: int
    tid: int
    start: float
    end: float
    duration: float
    energy: float
    cov: float

@dataclass
class Instance:
    sats: list
    targets: list
    ops: list
    horizon: float
    max_expected_coverage: float

def gen_instance(n_sat=5, n_targets=60, n_ops=800, seed=0):
    """
    Instancia sintética con geometría lineal:
    - Cada satélite se mueve en línea recta a velocidad constante sobre el AOI.
    - Ventanas analíticas: ||(p0 - g) + v*t|| <= FOV.
    - Energía = e0 + k * duración.
    - Cobertura opcional según centralidad (1..3).

    Args:
        n_sat (int): Número de satélites.
        n_targets (int): Número de objetivos.
        n_ops (int): Máximo de oportunidades a conservar.
        seed (int): Semilla de aleatoriedad.

    Returns:
        Instance: Instancia con satélites, objetivos, oportunidades y horizonte.
    """
    rng = np.random.default_rng(seed)

    # Configuración base
    H = 600.0            
    FOV = 1.0            
    MIN_DURATION = 1.0   
    E0 = 0.8
    K_ENERGY = 0.15
    AREA_MIN, AREA_MAX = -8.0, 8.0
    SPEED_MIN, SPEED_MAX = 0.03, 0.06

    # Satélites con posición inicial y velocidad
    sats = []
    sat_states = []
    for s in range(n_sat):
        e_max = rng.uniform(90, 140)
        t_max = rng.uniform(120, 180)
        sats.append(Satellite(s, e_max, t_max, 0.0))
        p0 = rng.uniform(AREA_MIN, AREA_MAX, size=2)
        speed = rng.uniform(SPEED_MIN, SPEED_MAX)
        theta = rng.uniform(0, 2 * np.pi)
        v = speed * np.array([np.cos(theta), np.sin(theta)])
        sat_states.append((p0, v))

    # Targets
    targets = []
    for t in range(n_targets):
        priority = rng.choice([1,2,3,4,5], p=[0.10,0.20,0.30,0.25,0.15])
        critical = (priority >= 4 and rng.random() < 0.75)
        min_obs = 1 if critical else 0
        x = rng.uniform(AREA_MIN, AREA_MAX)
        y = rng.uniform(AREA_MIN, AREA_MAX)
        targets.append(Target(t, float(priority), bool(critical), min_obs, float(x), float(y)))

    # Oportunidades
    ops = []
    best_cov_by_target = np.zeros(n_targets)
    oid = 0

    for sid, (p0, v) in enumerate(sat_states):
        vx, vy = v
        a = vx * vx + vy * vy
        if a < 1e-9:
            continue  # satélite casi estacionario
        for tid in range(n_targets):
            gx = targets[tid].x
            gy = targets[tid].y
            dx = p0[0] - gx
            dy = p0[1] - gy
            b = 2.0 * (vx * dx + vy * dy)
            c = dx * dx + dy * dy - FOV * FOV
            disc = b * b - 4.0 * a * c
            if disc < 0:
                continue
            sqrt_disc = float(np.sqrt(disc))
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            if t2 < 0 or t1 > H:
                continue
            t_start = max(0.0, t1)
            t_end = min(H, t2)
            duration = t_end - t_start
            if duration < MIN_DURATION:
                continue
            energy = float(E0 + K_ENERGY * duration)
            # distancia mínima dentro de la ventana
            t_star = np.clip(-b / (2.0 * a), t_start, t_end)
            r_star = np.array([dx + vx * t_star, dy + vy * t_star])
            d_min = float(np.linalg.norm(r_star))
            cov = 1.0  # cobertura uniforme por oportunidad

            best_cov_by_target[tid] = max(best_cov_by_target[tid], cov)
            ops.append(Opportunity(oid, sid, tid, t_start, t_end, duration, energy, cov))
            oid += 1

    # recortar al máximo solicitado si se generaron muchas oportunidades
    if len(ops) > n_ops:
        rng.shuffle(ops)
        ops = ops[:n_ops]

    # cobertura máxima esperada
    max_expected = 0.0
    for t in range(n_targets):
        max_expected += targets[t].priority * best_cov_by_target[t]

    return Instance(sats, targets, ops, H, max_expected)
