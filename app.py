"""Simulated Road-User Realism: Streamlit app.

    streamlit run app.py

Self-contained: the realism benchmark (features, classifier two-sample test,
control and the humanize intervention) runs live on CPU with requirements.txt.
"""
from dataclasses import astuple, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

COLS = ["source", "track_id", "t", "x", "y"]


# ---- features
@dataclass
class FeatureConfig:
    hz: float = 10.0
    window_s: float = 8.0
    stride_s: float = 4.0
    smooth_s: float = 0.5
    stop_speed: float = 0.2  # m/s
    min_path_m: float = 0.5  # windows shorter than this are standing-only and skipped for heading stats


FEATURES = [
    "speed_mean", "speed_std", "speed_p90", "speed_cv",
    "acc_abs_mean", "acc_abs_p90", "jerk_abs_mean",
    "stop_frac", "stop_go_transitions",
    "turn_rate_abs_mean", "turn_rate_abs_p90",
    "straightness", "speed_autocorr_1s",
]
# Route geometry depends on the map and on how agents are routed (navmesh corners vs.
# open plazas), not only on how a walker moves. Reporting both feature sets separates
# "the route looks fake" from "the walking looks fake".
GEOMETRY = ["straightness", "turn_rate_abs_mean", "turn_rate_abs_p90"]
MOTION = [f for f in FEATURES if f not in GEOMETRY]


def resample_track(tr: pd.DataFrame, cfg: FeatureConfig) -> np.ndarray | None:
    tr = tr.sort_values("t").drop_duplicates("t")
    if len(tr) < 3:
        return None
    t0, t1 = tr.t.iloc[0], tr.t.iloc[-1]
    ts = np.arange(t0, t1, 1.0 / cfg.hz)
    if len(ts) < 3:
        return None
    xy = np.stack([np.interp(ts, tr.t, tr.x), np.interp(ts, tr.t, tr.y)], axis=1)
    k = max(1, int(round(cfg.smooth_s * cfg.hz)))
    if k > 1 and len(xy) > k:
        kernel = np.ones(k) / k
        pad = k // 2
        padded = np.pad(xy, ((pad, k - 1 - pad), (0, 0)), mode="edge")
        xy = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)], axis=1)
    return xy


def kinematics(xy: np.ndarray, hz: float) -> dict[str, np.ndarray]:
    dt = 1.0 / hz
    v = np.diff(xy, axis=0) / dt
    speed = np.linalg.norm(v, axis=1)
    acc = np.diff(speed) / dt
    jerk = np.diff(acc) / dt
    heading = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    turn = np.diff(heading) / dt
    moving = (speed[1:] > 0.3) & (speed[:-1] > 0.3)  # heading is noise when standing still
    return {"speed": speed, "acc": acc, "jerk": jerk, "turn": turn[moving]}


def window_features(xy: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    k = kinematics(xy, cfg.hz)
    s, a, j, tr = k["speed"], k["acc"], k["jerk"], k["turn"]
    stopped = s < cfg.stop_speed
    path = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    disp = float(np.linalg.norm(xy[-1] - xy[0]))
    lag = int(cfg.hz)
    if len(s) > lag + 2 and s[:-lag].std() > 1e-6 and s[lag:].std() > 1e-6:
        ac = float(np.corrcoef(s[:-lag], s[lag:])[0, 1])
    else:
        ac = 1.0
    return {
        "speed_mean": float(s.mean()),
        "speed_std": float(s.std()),
        "speed_p90": float(np.percentile(s, 90)),
        "speed_cv": float(s.std() / (s.mean() + 1e-6)),
        "acc_abs_mean": float(np.abs(a).mean()),
        "acc_abs_p90": float(np.percentile(np.abs(a), 90)),
        "jerk_abs_mean": float(np.abs(j).mean()),
        "stop_frac": float(stopped.mean()),
        "stop_go_transitions": float(np.sum(stopped[1:] != stopped[:-1])),
        "turn_rate_abs_mean": float(np.abs(tr).mean()) if len(tr) else 0.0,
        "turn_rate_abs_p90": float(np.percentile(np.abs(tr), 90)) if len(tr) else 0.0,
        "straightness": disp / path if path > cfg.min_path_m else 1.0,
        "speed_autocorr_1s": ac if np.isfinite(ac) else 1.0,
    }


def windows_from_tracks(df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """One row per window: source, track_id, start index, then FEATURES."""
    n, step = int(cfg.window_s * cfg.hz), int(cfg.stride_s * cfg.hz)
    rows = []
    for (src, tid), tr in df.groupby(["source", "track_id"], sort=False):
        xy = resample_track(tr, cfg)
        if xy is None or len(xy) < n:
            continue
        for s in range(0, len(xy) - n + 1, step):
            f = window_features(xy[s : s + n], cfg)
            rows.append({"source": src, "track_id": tid, "start": s, **f})
    return pd.DataFrame(rows, columns=["source", "track_id", "start", *FEATURES])


def pointwise(df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """Per-time-step speed / |acc| / |turn rate| samples, for distribution plots."""
    out = []
    for (src, tid), tr in df.groupby(["source", "track_id"], sort=False):
        xy = resample_track(tr, cfg)
        if xy is None or len(xy) < 4:
            continue
        k = kinematics(xy, cfg.hz)
        out.append(pd.DataFrame({"source": src, "kind": "speed", "value": k["speed"]}))
        out.append(pd.DataFrame({"source": src, "kind": "acc_abs", "value": np.abs(k["acc"])}))
        out.append(pd.DataFrame({"source": src, "kind": "turn_rate_abs", "value": np.abs(k["turn"])}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["source", "kind", "value"])


# ---- realism metrics
def feature_distances(ref: pd.DataFrame, other: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    rows = []
    for f in features:
        a, b = ref[f].to_numpy(), other[f].to_numpy()
        sd = a.std() or 1.0
        rows.append({"feature": f, "ref_mean": a.mean(), "other_mean": b.mean(), "w1_std_units": wasserstein_distance(a, b) / sd})
    return pd.DataFrame(rows).sort_values("w1_std_units", ascending=False, ignore_index=True)


def _balance(a: pd.DataFrame, b: pd.DataFrame, rng: np.random.Generator):
    n = min(len(a), len(b))
    return a.iloc[rng.choice(len(a), n, replace=False)], b.iloc[rng.choice(len(b), n, replace=False)]


def c2st(a: pd.DataFrame, b: pd.DataFrame, features=FEATURES, folds: int = 5, seed: int = 0,
         importance: bool = True) -> dict:
    """Classifier two-sample test between window sets a (label 0) and b (label 1)."""
    rng = np.random.default_rng(seed)
    a, b = _balance(a, b, rng)
    X = pd.concat([a[features], b[features]], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    groups = pd.concat([a.track_id, b.track_id], ignore_index=True).astype("category").cat.codes.to_numpy()
    n_groups = len(np.unique(groups))
    folds = min(folds, n_groups)
    aucs, imps = [], []
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
        clf.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X.iloc[te])[:, 1]))
        if importance:
            pi = permutation_importance(clf, X.iloc[te], y[te], scoring="roc_auc", n_repeats=5, random_state=seed)
            imps.append(pi.importances_mean)
    out = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "folds": len(aucs), "n_per_class": len(a)}
    if importance and imps:
        imp = pd.Series(np.mean(imps, axis=0), index=features).sort_values(ascending=False)
        out["importance"] = imp
    return out


def real_vs_real_control(real: pd.DataFrame, seed: int = 0, **kw) -> dict:
    """Split real tracks in two halves and run the same test. Should give AUC ≈ 0.5;
    if it does not, the features pick up something other than realism (e.g. scene identity)."""
    rng = np.random.default_rng(seed)
    tracks = np.array(real.track_id.unique(), dtype=object)
    rng.shuffle(tracks)
    half = set(tracks[: len(tracks) // 2])
    m = real.track_id.isin(half)
    return c2st(real[m], real[~m], seed=seed, importance=False, **kw)


# ---- humanize intervention
@dataclass
class HumanizeParams:
    speed_sigma: float = 0.25     # relative std of speed multiplier
    speed_tau: float = 2.0        # s, correlation time of speed changes
    stop_rate: float = 0.03       # stops per second
    stop_mean_s: float = 1.5      # mean stop duration
    sway_sigma: float = 0.08      # m, lateral sway amplitude
    sway_tau: float = 1.5         # s
    obs_noise: float = 0.02       # m, position noise like a real tracker/detector adds
    corner_s: float = 1.0         # s, rounds off sharp navmesh corners before re-timing


def _ou(n: int, dt: float, tau: float, sigma: float, rng) -> np.ndarray:
    x = np.zeros(n)
    a = np.exp(-dt / tau)
    b = sigma * np.sqrt(1 - a * a)
    for i in range(1, n):
        x[i] = a * x[i - 1] + b * rng.standard_normal()
    return x


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) <= k:
        return x
    pad = k // 2
    return np.convolve(np.pad(x, (pad, k - 1 - pad), mode="edge"), np.ones(k) / k, mode="valid")


def humanize_xy(xy: np.ndarray, hz: float, p: HumanizeParams, rng) -> np.ndarray:
    dt = 1.0 / hz
    xy = np.stack([_smooth(xy[:, i], int(p.corner_s * hz)) for i in range(2)], axis=1)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    if arc[-1] < 1e-6:
        return xy.copy()
    n = len(xy)
    base_speed = arc[-1] / ((n - 1) * dt)
    mult = np.clip(1 + _ou(n, dt, p.speed_tau, p.speed_sigma, rng), 0.2, 2.0)
    i, moving = 0, np.ones(n, bool)
    while i < n:
        gap = rng.exponential(1 / p.stop_rate) if p.stop_rate > 0 else np.inf
        i += int(gap * hz)
        if i >= n:
            break
        dur = int(rng.exponential(p.stop_mean_s) * hz) + 1
        moving[i : i + dur] = False
        i += dur
    # people decelerate into and accelerate out of a stop over ~0.6 s rather than instantly
    moving_f = _smooth(moving.astype(float), int(0.6 * hz))
    step = base_speed * mult * moving_f * dt
    s = np.clip(np.r_[0, np.cumsum(step[:-1])], 0, arc[-1])
    new = np.stack([np.interp(s, arc, xy[:, 0]), np.interp(s, arc, xy[:, 1])], axis=1)
    # lateral sway: a smoothed OU offset, so the path weaves gently instead of jittering
    tangent = np.stack([np.interp(s, arc, np.gradient(xy[:, i])) for i in range(2)], axis=1)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    perp = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1) / np.maximum(norm, 1e-9)
    raw = _smooth(_ou(n, dt, p.sway_tau, 1.0, rng), int(p.sway_tau * hz))
    sway = (raw / (raw.std() + 1e-9) * p.sway_sigma * moving_f)[:, None]
    return new + perp * sway + rng.normal(0, p.obs_noise, new.shape)


def humanize(df: pd.DataFrame, p: HumanizeParams, cfg: FeatureConfig = FeatureConfig(), seed: int = 0,
             source: str = "sim_humanized") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for tid, tr in df.groupby("track_id", sort=False):
        xy = resample_track(tr, cfg)
        if xy is None:
            continue
        h = humanize_xy(xy, cfg.hz, p, rng)
        t0 = tr.t.min()
        out.append(pd.DataFrame({"source": source, "track_id": tid, "t": t0 + np.arange(len(h)) / cfg.hz,
                                 "x": h[:, 0], "y": h[:, 1]}))
    return pd.concat(out, ignore_index=True)


# ---- synthetic stand-ins for the default demo data
# real: walkers with individual preferred speeds, OU speed fluctuations, occasional stops,
#       mild heading wander and tracker noise (roughly human-like)
# sim:  constant speed along straight segments with sharp turns (roughly navmesh-like)
FPS = 25


def _real_track(rng, seconds):
    n = int(seconds * FPS)
    dt = 1 / FPS
    v0 = rng.normal(1.35, 0.2)
    v = np.empty(n)
    x = 0.0
    for i in range(n):
        x = x * np.exp(-dt / 2) + 0.25 * np.sqrt(1 - np.exp(-2 * dt / 2)) * rng.standard_normal()
        v[i] = max(0.0, v0 * (1 + x))
    i = 0
    while i < n:
        i += int(rng.exponential(15) * FPS)
        d = int(rng.exponential(1.5) * FPS)
        v[i : i + d] = 0
        i += d
    h = np.cumsum(rng.normal(0, 0.03, n)) + rng.uniform(0, 2 * np.pi)
    xy = np.cumsum(np.stack([v * np.cos(h), v * np.sin(h)], 1) * dt, axis=0)
    return xy + rng.normal(0, 0.02, xy.shape)  # tracker noise


def _sim_track(rng, seconds):
    n = int(seconds * FPS)
    dt = 1 / FPS
    v = rng.choice([1.2, 1.4])
    h = rng.uniform(0, 2 * np.pi)
    xy = np.zeros((n, 2))
    for i in range(1, n):
        if rng.random() < 1 / (6 * FPS):
            h += rng.choice([-1, 1]) * np.pi / 2
        xy[i] = xy[i - 1] + v * dt * np.array([np.cos(h), np.sin(h)])
    return xy


def synthetic_tracks(n_real: int = 60, n_sim: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    real, sim = [], []
    for tid in range(n_real):
        xy = _real_track(rng, rng.uniform(20, 40))
        real.append(pd.DataFrame({"source": "real", "track_id": f"real_{tid}", "t": np.arange(len(xy)) / FPS,
                                  "x": xy[:, 0], "y": xy[:, 1]}))
    for tid in range(n_sim):
        xy = _sim_track(rng, rng.uniform(20, 40))
        sim.append(pd.DataFrame({"source": "sim", "track_id": f"sim_{tid}", "t": np.arange(len(xy)) / FPS,
                                 "x": xy[:, 0], "y": xy[:, 1]}))
    return pd.concat(real, ignore_index=True), pd.concat(sim, ignore_index=True)


GITHUB = "https://github.com/Victortaiwo57/sim2real-road-users"
BLUE, RED, AMBER = "#2a6f97", "#d1495b", "#edae49"
cfg = FeatureConfig()

st.set_page_config(page_title="Simulated Road-User Realism", page_icon="🧪", layout="wide")

st.caption("Simulation · making simulated road users more realistic")
st.title("🧪 How realistic are simulated pedestrians?")
st.markdown(
    "A measurable test of the **sim-to-real gap**: can a classifier tell pedestrians from a driving simulator "
    "(**CARLA**) apart from real pedestrians filmed at intersections (**inD**)? Which behaviours give the "
    "simulation away, and does a simple change to the simulated walkers close the gap?"
)

tab_over, tab_demo, tab_res, tab_method = st.tabs(["Overview", "Live benchmark", "Research plan", "Method"])

with tab_over:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Why it matters")
        st.markdown(
            "- Automated vehicles and driver-assistance systems are increasingly **tested in simulation**, because "
            "rare, dangerous situations are slow and risky to test on real roads.\n"
            "- Those tests are only as good as the simulated people in them. Simulated pedestrians often walk at a "
            "constant speed along clean paths, while real people hesitate, stop, speed up and weave.\n"
            "- If simulated road users are too predictable, a system can **pass in simulation and fail on a real "
            "street**, which is exactly the failure simulation is meant to prevent.\n"
            "- You cannot improve what you cannot measure. This project builds a realism metric with a control and "
            "a held-out evaluation, and shows what it does and does not capture."
        )
    with c2:
        st.subheader("Where it could be used")
        st.markdown(
            "- **AV and ADAS validation teams:** checking that the pedestrians in test scenarios behave like real ones.\n"
            "- **Simulator developers (CARLA, commercial tools):** a regression test for pedestrian-behaviour models.\n"
            "- **Researchers building behaviour models:** one benchmark to compare learned pedestrian models fairly.\n"
            "- **Safety assessors and regulators:** evidence on whether simulation-based test results transfer.\n"
            "- **Cities with very different traffic:** measuring how far simulators are from local road-user "
            "behaviour, e.g. Lagos, before trusting them."
        )
    st.subheader("Research questions")
    st.markdown(
        "1. How easily can a classifier separate CARLA pedestrian trajectories from real inD trajectories?\n"
        "2. Which behaviours (speed changes, stops, acceleration, path shape) give the simulation away?\n"
        "3. How much of the gap is *behaviour*, and how much is *measurement noise* or *map/route geometry*?\n"
        "4. Does a simple, explicit 'humanize' change close the gap on held-out real data?"
    )
    st.info("The **Live benchmark** tab runs the real metric in your browser. Move the sliders, or upload your own "
            "trajectories.")

with tab_demo:
    st.markdown(
        "The benchmark runs live below. By default it uses **synthetic stand-ins** (human-like walkers vs. "
        "navmesh-like walkers), clearly not real data. Upload your own CSVs (`track_id, t, x, y` in seconds and "
        "metres) to run it on real trajectories. Move the sliders to **humanize** the simulated walkers and see whether "
        "the classifier can still tell them apart."
    )

    @st.cache_data(show_spinner=False)
    def demo_tracks():
        return synthetic_tracks(60, 60, seed=0)

    def read_upload(f, source):
        df = pd.read_csv(f)
        missing = {"track_id", "t", "x", "y"} - set(df.columns)
        if missing:
            st.error(f"{f.name}: missing columns {sorted(missing)}")
            st.stop()
        return df.assign(source=source, track_id=source + "_" + df.track_id.astype(str))[COLS]

    with st.expander("Use your own trajectories"):
        u1, u2 = st.columns(2)
        up_real = u1.file_uploader("Real trajectories CSV", type="csv")
        up_sim = u2.file_uploader("Simulated trajectories CSV", type="csv")
    real, sim = demo_tracks()
    if up_real is not None:
        real = read_upload(up_real, "real")
    if up_sim is not None:
        sim = read_upload(up_sim, "sim")

    st.markdown("**Humanize the simulated walkers**")
    h1, h2, h3, h4, h5 = st.columns(5)
    p = HumanizeParams(
        speed_sigma=h1.slider("Speed variability", 0.0, 0.5, 0.25, 0.05),
        stop_rate=h2.slider("Stops per minute", 0.0, 6.0, 1.8, 0.2) / 60,
        sway_sigma=h3.slider("Lateral sway (m)", 0.0, 0.2, 0.08, 0.01),
        obs_noise=h4.slider("Tracker noise (m)", 0.0, 0.06, 0.02, 0.005),
        corner_s=h5.slider("Corner rounding (s)", 0.0, 3.0, 1.0, 0.25),
    )

    @st.cache_data(show_spinner="Running classifier two-sample tests…")
    def bench(real, sim, p_tuple):
        hum = humanize(sim, HumanizeParams(*p_tuple), cfg, seed=0)
        rw, sw, hw = (windows_from_tracks(x, cfg) for x in (real, sim, hum))
        auc = {}
        for label, feats in [("motion", MOTION), ("all", FEATURES)]:
            auc[label] = {
                "control": real_vs_real_control(rw, features=feats)["auc_mean"],
                "sim": c2st(rw, sw, features=feats, importance=False)["auc_mean"],
                "hum": c2st(rw, hw, features=feats, importance=False)["auc_mean"],
            }
        imp = c2st(rw, hw)["importance"]
        pts = pointwise(pd.concat([real, sim, hum], ignore_index=True), cfg)
        return auc, imp, feature_distances(rw, sw), feature_distances(rw, hw), pts

    auc, imp, d_sim, d_hum, pts = bench(real, sim, astuple(p))
    mo, al = auc["motion"], auc["all"]

    st.markdown("**Can a classifier tell them apart?** Motion dynamics only (speed, acceleration, stops)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Real vs real (control)", f"{mo['control']:.2f}", help="Two halves of the real data. Should be near 0.5.")
    m2.metric("Real vs simulated", f"{mo['sim']:.2f}")
    m3.metric("Real vs humanized simulation", f"{mo['hum']:.2f}", delta=f"{mo['hum'] - mo['sim']:+.2f}",
              delta_color="inverse")
    st.caption(
        f"With route geometry included (straightness, turn rates) the AUCs are {al['control']:.2f} / {al['sim']:.2f} / "
        f"{al['hum']:.2f}. ROC-AUC: 0.5 = indistinguishable from real, 1.0 = trivially separable. Cross-validation is "
        "grouped by trajectory, so one walker never sits in both train and test."
    )

    kinds = [("speed", "speed (m/s)", 3.0), ("acc_abs", "|acceleration| (m/s²)", 3.0),
             ("turn_rate_abs", "|turn rate| (rad/s)", 2.0)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.8))
    for ax, (k, lab, hi) in zip(axes, kinds):
        bins = np.linspace(0, hi, 50)
        for src, col in [("real", BLUE), ("sim", RED), ("sim_humanized", AMBER)]:
            v = pts[(pts.kind == k) & (pts.source == src)].value.clip(upper=hi)
            if len(v):
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8, color=col, label=src.replace("_", " "))
        real_v = pts[(pts.kind == k) & (pts.source == "real")].value.clip(upper=hi)
        if len(real_v):
            ax.set_ylim(0, np.histogram(real_v, bins=bins, density=True)[0].max() * 2.2)
        ax.set_xlabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

    left, right = st.columns(2)
    with left:
        st.markdown("**What still gives the humanized walkers away** (drop in AUC when a feature is shuffled)")
        st.bar_chart(imp.head(6), horizontal=True, color=BLUE)
    with right:
        st.markdown("**Per-feature gaps** (Wasserstein distance in real-data std units)")
        both = d_sim[["feature", "w1_std_units"]].rename(columns={"w1_std_units": "sim"}).merge(
            d_hum[["feature", "w1_std_units"]].rename(columns={"w1_std_units": "humanized"}), on="feature")
        st.dataframe(both.head(8).round(2), hide_index=True, width="stretch")
    st.markdown(
        "**What to try:** set *Tracker noise* to 0 and the motion-dynamics gap comes back, even with realistic speed "
        "changes and stops. Real trajectories come from a tracker and carry noise; simulator ground truth does not, so "
        "a realism metric that ignores this partly measures **sensor noise rather than behaviour**. Also notice that "
        "route geometry still separates the two sets after humanizing. That gap comes from the map and routing, not "
        "from how the walkers move, which is why the benchmark reports both feature sets."
    )

with tab_res:
    st.markdown(
        "**Status:** the live benchmark above runs the real metric on synthetic stand-ins (or your uploaded data). "
        "The full experiment compares **CARLA** pedestrians (recorded from AI walkers in several towns at 25 Hz) with "
        "real pedestrians from **inD** (drone recordings at German intersections). These are the comparisons it will "
        "report. No numbers are shown until they exist."
    )
    st.dataframe(pd.DataFrame({
        "comparison": ["real vs real (control)", "real vs CARLA", "real vs CARLA + humanize"],
        "expected": ["≈ 0.5", "high", "lower, if humanize works"],
        "AUC (all features)": ["–"] * 3, "AUC (motion dynamics only)": ["–"] * 3,
    }), hide_index=True, width="stretch")
    st.markdown(
        "**Protocol:** humanize parameters fitted on one half of the real tracks and evaluated on the other half only; "
        "folds grouped by trajectory; mean ± std over folds and seeds; an ablation without the tracker-noise term; "
        "results per CARLA town and per inD intersection to expose map effects."
    )

with tab_method:
    st.code(
        "inD tracks (25 Hz) ──┐                                            ┌─► per-feature Wasserstein distance\n"
        "                     ├─► resample 10 Hz ─► smooth ─► 8 s windows ─┼─► classifier two-sample test (AUC)\n"
        f"CARLA walkers ───────┘      {len(FEATURES)} kinematic features              └─► permutation importance",
        language=None,
    )
    st.markdown(
        "- **Same processing for both sources:** only positions are used; speed, acceleration, jerk and turn rate are "
        "recomputed identically, so dataset preprocessing can't leak into the comparison.\n"
        "- **Classifier two-sample test:** a gradient-boosted classifier tries to separate real from simulated 8 s "
        "windows; folds are grouped by trajectory.\n"
        "- **Control:** two halves of the real data should be indistinguishable (AUC ≈ 0.5). If not, the features are "
        "picking up something other than realism.\n"
        "- **Two feature sets:** all features, and motion dynamics only (route geometry removed, because it depends on "
        "the map as much as on the walker).\n"
        "- **Intervention:** `humanize` adds speed modulation, smooth stops, lateral sway, corner rounding and "
        "tracker-like noise. Its parameters are fitted on one half of the real tracks and evaluated on the other half.\n"
        "- **Simulation:** AI walkers recorded from a CARLA server in synchronous mode at 25 Hz.\n"
        "- **Limitations:** kinematics only (no interaction with vehicles yet); maps differ between datasets; "
        "the classifier test says *that* distributions differ, not whether it matters for a downstream task."
    )
    st.markdown(f"Code and README: [{GITHUB.replace('https://', '')}]({GITHUB})")

st.divider()
st.caption("Victor Taiwo · Lagos, Nigeria · research prototype, not for safety-critical use.")
