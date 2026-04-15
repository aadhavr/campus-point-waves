# model.py
# Fits on paired forecast+buoy observations, predicts Hs at Campus Point.
# Features: mop_raw_Hs, harvest_Hs, harvest_Dp (sin/cos), harvest_Tp (when available)
# Target: buoy_Hs

import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

MIN_ROWS  = 30    # minimum paired obs before model is used
MAX_ROWS  = 2000  # rolling window cap — older data gets dropped
ALPHA     = 1.0   # Ridge regularization strength

STATE_PATH = 'model_state.json'
SEED_PATH  = 'val_check_1/validation_seed.csv'


def _build_features(df):
    """
    Build feature matrix from a dataframe with columns:
    Hs_mop_raw, harvest_Hs, harvest_Dp, and optionally harvest_Tp.
    Returns numpy array X.
    """
    dp_rad = np.deg2rad(df['harvest_Dp'].values)
    X = np.column_stack([
        df['Hs_mop_raw'].values,
        df['harvest_Hs'].values,
        np.sin(dp_rad),
        np.cos(dp_rad),
    ])
    # add Tp if present and not all NaN
    if 'harvest_Tp' in df.columns and df['harvest_Tp'].notna().sum() > MIN_ROWS:
        X = np.column_stack([X, df['harvest_Tp'].values])
    return X


def _ridge_fit(X, y, alpha=ALPHA):
    """
    Closed-form Ridge regression: w = (X'X + αI)^-1 X'y
    Returns weight vector w.
    """
    n_feat = X.shape[1]
    A = X.T @ X + alpha * np.eye(n_feat)
    b = X.T @ y
    return np.linalg.solve(A, b)


class WaveModel:
    def __init__(self):
        self.weights    = None
        self.n_features = None
        self.n_obs      = 0
        self.last_fit   = None
        self.use_tp     = False

    def fit(self, df):
        """
        Fit on a dataframe of paired observations.
        Drops rows with any NaN in required columns.
        """
        required = ['Hs_mop_raw', 'harvest_Hs', 'harvest_Dp', 'buoy_Hs']
        df = df.dropna(subset=required).copy()

        if len(df) < MIN_ROWS:
            print(f"WaveModel: only {len(df)} rows, need {MIN_ROWS} — not fitting")
            return False

        # cap to rolling window
        if len(df) > MAX_ROWS:
            df = df.tail(MAX_ROWS)

        X = _build_features(df)
        y = df['buoy_Hs'].values

        self.weights    = _ridge_fit(X, y)
        self.n_features = X.shape[1]
        self.n_obs      = len(df)
        self.last_fit   = datetime.now(timezone.utc).isoformat()
        self.use_tp     = (self.n_features == 5)

        print(f"WaveModel: fit on {self.n_obs} obs, features={self.n_features}, "
              f"weights={[round(w,4) for w in self.weights]}")
        return True

    def predict(self, mop_raw, harvest_Hs, harvest_Dp, harvest_Tp=None):
        """
        Predict Campus Point Hs for a single observation.
        Returns float or None if model not ready.
        """
        if self.weights is None:
            return None

        dp_rad = np.deg2rad(harvest_Dp)
        x = [mop_raw, harvest_Hs, np.sin(dp_rad), np.cos(dp_rad)]

        if self.use_tp:
            if harvest_Tp is None:
                return None
            x.append(harvest_Tp)

        x = np.array(x)
        if len(x) != self.n_features:
            return None

        pred = float(x @ self.weights)
        return round(max(pred, 0.0), 3)  # clamp to non-negative

    def save(self, path=STATE_PATH):
        state = {
            'weights':    self.weights.tolist() if self.weights is not None else None,
            'n_features': self.n_features,
            'n_obs':      self.n_obs,
            'last_fit':   self.last_fit,
            'use_tp':     self.use_tp,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"WaveModel: saved to {path}")

    def load(self, path=STATE_PATH):
        try:
            with open(path) as f:
                state = json.load(f)
            if state['weights'] is None:
                return False
            self.weights    = np.array(state['weights'])
            self.n_features = state['n_features']
            self.n_obs      = state['n_obs']
            self.last_fit   = state['last_fit']
            self.use_tp     = state['use_tp']
            print(f"WaveModel: loaded from {path}, n_obs={self.n_obs}, last_fit={self.last_fit}")
            return True
        except FileNotFoundError:
            print(f"WaveModel: no state file at {path}, will fit from scratch")
            return False
        except Exception as e:
            print(f"WaveModel: load error ({e}), will fit from scratch")
            return False


def load_training_data(log_path='forecast_log.csv', seed_path=SEED_PATH):
    """
    Load all available paired observations for training.
    Combines validation seed with live forecast log rows that have buoy_Hs.
    Returns a single dataframe sorted by time.
    """
    frames = []

    # seed data from val_check_1
    try:
        seed = pd.read_csv(seed_path)
        frames.append(seed)
        print(f"WaveModel: loaded {len(seed)} seed rows from {seed_path}")
    except FileNotFoundError:
        print(f"WaveModel: no seed file at {seed_path}")

    # live log rows with buoy obs
    try:
        live = pd.read_csv(log_path)
        live = live[live['buoy_Hs'].notna()].copy()
        if len(live) > 0:
            # rename to match seed schema
            live = live.rename(columns={'harvest_Dp': 'harvest_Dp'})
            frames.append(live[['mop_time', 'Hs_mop_raw', 'harvest_Hs',
                                 'harvest_Dp', 'buoy_Hs', 'buoy_Tp']])
            print(f"WaveModel: loaded {len(live)} live rows from {log_path}")
    except FileNotFoundError:
        print(f"WaveModel: no log file at {log_path}")
    except Exception as e:
        print(f"WaveModel: live log error ({e})")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if 'mop_time' in df.columns:
        df = df.sort_values('mop_time').reset_index(drop=True)
    return df