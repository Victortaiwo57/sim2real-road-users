# How Realistic Are Simulated Pedestrians? Measuring the Sim-to-Real Gap

**Live app:** https://sim2real-road-users.streamlit.app

This project builds a measurable test of how realistic simulated pedestrians are. It asks whether a classifier can tell pedestrians from a driving simulator (**CARLA**) apart from real pedestrians filmed at intersections (**inD**), which behaviours give the simulation away, and whether a simple, explicit change to the simulated walkers closes the gap on held-out real data.

The repository is a Streamlit app in which the benchmark runs live in the browser on demo data or on your own uploaded trajectories, with a README that sets out the full CARLA vs inD experiment.

---

## Why this project is important

- Road traffic kills about **1.19 million people a year**, and more than half are pedestrians, cyclists and motorcyclists ([WHO, 2023](https://www.who.int/publications/i/item/9789240086517)). The systems meant to protect them are increasingly **validated in simulation**, because rare, dangerous situations are slow and risky to test on real roads.
- Simulation results are only as good as the simulated people. Simulated pedestrians often walk at constant speed along clean navmesh paths, while real people hesitate, stop, speed up, weave and cut corners.
- If simulated road users are too predictable, a system can **pass in simulation and fail on a real street**, which is the very failure simulation is meant to prevent.
- Realism is often judged by eye. This project turns it into a **number with a control and a held-out evaluation**, and shows what the number does and does not capture.
- A development finding with real methodological weight: much of an apparent sim-to-real gap can come from **tracker noise** in real data (simulator ground truth is noise-free) and from **map/route geometry**, not from behaviour. A realism metric that ignores this can mislead.

## Use cases

| Who | How this helps |
|---|---|
| AV and ADAS validation teams | Check that pedestrians in test scenarios behave like real ones before trusting test results |
| Simulator developers (CARLA, commercial tools) | A regression test for pedestrian-behaviour models: did the new version get more or less realistic? |
| Researchers building behaviour models | One benchmark to compare learned pedestrian models fairly, with a built-in control |
| Safety assessors and regulators | Evidence on whether simulation-based results are likely to transfer to real roads |
| Cities with very different traffic (e.g. Lagos) | Measure how far a simulator is from local road-user behaviour before relying on it |
| Anyone with trajectory data | Upload two CSVs in the app and compare them instantly |

## Research questions

1. How easily can a classifier separate CARLA pedestrian trajectories from real inD trajectories?
2. Which behaviours (speed variation, stops, acceleration, path shape) give the simulation away?
3. How much of the gap is **behaviour**, and how much is **measurement noise** or **route geometry**?
4. Does a simple, explicit "humanize" intervention close the gap on **held-out** real data?

## What the Streamlit app shows

| Tab | Contents |
|---|---|
| **Overview** | Why realism matters, who it is useful for, and the research questions |
| **Live benchmark** | The actual realism metric running in the browser. Sliders humanize the simulated walkers (speed variability, stops, sway, tracker noise, corner rounding), and the app shows classifier AUCs with a real-vs-real control, distribution plots, the features that still give the simulation away, and per-feature gaps. **Upload your own real and simulated CSVs** (`track_id, t, x, y`) to run it on your data. The default data are clearly labelled synthetic stand-ins. |
| **Research plan** | The CARLA vs inD comparisons the full experiment will report, and the protocol |
| **Method** | Pipeline, controls and limitations |

The app runs on CPU with light dependencies, so it fits Streamlit Community Cloud's free tier.

## Method

```
inD tracks (drone, 25 Hz) ─┐                                              ┌─► per-feature Wasserstein distance
                            ├─► resample 10 Hz ─► smooth ─► 8 s windows ─► ┼─► classifier two-sample test (AUC)
CARLA walkers (25 Hz)  ─────┘      13 kinematic features per window        └─► permutation importance
```

- **Same processing for both sources.** Only positions are used. Speed, acceleration, jerk and turn rate are recomputed identically, so dataset-specific preprocessing cannot leak into the comparison.
- **Features per 8 s window:** speed (mean, std, p90, coefficient of variation), |acceleration| (mean, p90), |jerk|, stop fraction, stop–go transitions, |turn rate| (mean, p90), path straightness, 1 s speed autocorrelation.
- **Classifier two-sample test (C2ST):** a gradient-boosted classifier tries to separate real from simulated windows. AUC 0.5 means indistinguishable. Folds are **grouped by trajectory**, so windows from one walker never appear on both sides of a split.
- **Control:** the same test between two random halves of the real tracks should give AUC ≈ 0.5. If it doesn't, the features are picking up something other than realism.
- **Two feature sets:** all features, and **motion dynamics only** (route geometry removed), separating "the route looks fake" (map and routing) from "the walking looks fake" (behaviour).
- **Intervention (`humanize`):** OU speed modulation, short stops with smooth deceleration, gentle lateral sway, corner rounding, and tracker-like position noise. Parameters are fitted by random search on **one half** of the real tracks; every reported number uses the **other half**.

## Results

> The live benchmark runs the real metric on synthetic stand-ins or uploaded data. The CARLA vs inD experiment is the next step, and this table will be filled when it is run. No numbers here are invented.

| comparison | AUC (all features) | AUC (motion dynamics only) |
|---|---|---|
| real vs real (control) | | |
| real vs CARLA | | |
| real vs CARLA + humanize | | |

**What gives the simulation away:** _top features by permutation importance, with a sentence on each_.

## Environment

| Component | Version / requirement |
|---|---|
| Python | 3.10 or 3.11 |
| OS | Linux, macOS or Windows |
| **Streamlit app + benchmark** | CPU only. `requirements.txt`: streamlit ≥ 1.50, numpy, pandas, scipy, scikit-learn, matplotlib |
| Network | Only needed to install packages; the app downloads nothing at runtime |
| Hardware | The benchmark runs on a laptop CPU in seconds |
| **Simulation (planned experiment)** | CARLA 0.9.x server plus the matching `carla` Python package. Needs a GPU machine; CARLA does not run on Colab or Streamlit Cloud |
| Datasets | [inD](https://levelxdata.com/ind-dataset/) (free for non-commercial research, on request; check its licence terms). Any trajectory data with `track_id, t, x, y` can be uploaded to the app. |

## Run the app locally

```bash
git clone https://github.com/Victortaiwo57/sim2real-road-users.git
cd sim2real-road-users
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt` and `README.md` to a public GitHub repository named `sim2real-road-users`.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → choose the repo, branch `main`, main file **`app.py`**.
3. Under **Advanced settings**, choose Python 3.11 and set the app URL to `sim2real-road-users`.
4. Deploy. The app needs no GPU and no PyTorch.

## How the full experiment will be run

1. **Simulated data:** run a CARLA 0.9.x server in synchronous mode at 25 Hz, spawn about 80 AI walkers per town with random navigation goals, and record their positions for 5 minutes in several towns (e.g. Town03, Town10).
2. **Real data:** pedestrian tracks from inD (25 Hz drone recordings at four German intersections).
3. **Export both** as CSV with `track_id, t, x, y` (seconds and metres) and run the same benchmark that the app runs: resample to 10 Hz, smooth, cut into 8 s windows, and compute 13 kinematic features.
4. **Tests:** classifier two-sample test (folds grouped by trajectory) on all features and on motion dynamics only, with the real-vs-real control alongside.
5. **Intervention:** fit the humanize parameters by random search on one half of the real tracks; report every number on the other half.
6. **Ablations:** without the tracker-noise term; per CARLA town and per inD intersection.

## Evaluation plan

- Always report the real-vs-real control next to the sim comparisons. A sim AUC is only meaningful relative to it.
- Report both feature sets (all, and motion dynamics only), with mean ± std over folds and at least 3 seeds.
- Fit any intervention on one half of the real tracks and evaluate on the other half only.
- Run an ablation of the observation-noise term to quantify how much of the gap is sensor noise.
- Use several CARLA towns and several inD intersections, and report per-location results to expose layout effects.

## Limitations

- Kinematics only. Interaction with vehicles and other pedestrians (gap acceptance, yielding, group walking) is not measured yet, and that is probably where realism matters most.
- inD intersections and CARLA towns have different layouts, so route geometry is partly a property of the map.
- The classifier test shows *that* distributions differ, not whether the difference matters for a downstream task.
- `humanize` is a statistical patch, not a behaviour model.
- The live benchmark's default data are synthetic; they illustrate the method and are not evidence about CARLA.
- Research prototype, not for safety-critical use. Check dataset and simulator licence terms before any commercial use.

## What I would do next

- Add interaction features: minimum distance to vehicles, time-to-collision at crossing start, kerb waiting time.
- Extend to cyclists and vehicles (inD includes both).
- Replace `humanize` with a learned trajectory model trained on inD and compare it on the same benchmark.
- Check whether a crossing-intention model trained in simulation degrades on real data in proportion to the measured AUC.

## Repository layout

```
app.py             the Streamlit app (self-contained: demo code, charts and text)
requirements.txt   Python packages for the app
README.md          this file
```

## References

- Bock et al. *The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections.* IEEE IV, 2020.
- Dosovitskiy et al. *CARLA: An Open Urban Driving Simulator.* CoRL, 2017.
- Lopez-Paz, Oquab. *Revisiting Classifier Two-Sample Tests.* ICLR, 2017.
- World Health Organization. *Global status report on road safety 2023.*

## Related projects

- [RoadMind Lite](https://github.com/Victortaiwo57/roadmind-lite): road-object detection baseline (YOLO + Streamlit).
- [crossing-intent-fm](https://github.com/Victortaiwo57/crossing-intent-fm): will this pedestrian cross?
- [driver-attention-fm](https://github.com/Victortaiwo57/driver-attention-fm): where do drivers look?

## Author

**Victor Taiwo**, Lagos, Nigeria · [LinkedIn](https://www.linkedin.com/in/victor-taiwo-076094167/) · [GitHub](https://github.com/Victortaiwo57)

MIT License
