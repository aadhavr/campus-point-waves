# forecast.py
# Pulls latest MOP and Harvest data, applies bias correction,
# outputs a corrected wave forecast for Campus Point.
# Runs hourly via GitHub Actions.

import json
import sys
import numpy as np
import pandas as pd
import xarray as xr
from model import WaveModel, load_training_data, STATE_PATH

# --- Load correction model ---
with open('campus_point_correction.json') as f:
    model = json.load(f)

# --- Correction function ---
def correct_forecast(Hs_mop, Dp_harvest, model):
    bins    = model['bin_edges']
    factors = model['scale_factors']
    conf    = model['confidence']

    for name, (lo, hi) in bins.items():
        if lo <= Dp_harvest < hi:
            return {
                'Hs_corrected': round(Hs_mop * factors[name], 3),
                'confidence':   conf[name]['flag'],
                'R2':           conf[name]['R2'],
                'bin':          name,
                'scale_factor': factors[name],
            }

    return {
        'Hs_corrected': round(Hs_mop * factors['global'], 3),
        'confidence':   'LOW',
        'R2':           None,
        'bin':          'global fallback',
        'scale_factor': factors['global'],
    }

def fetch_spot_observation(hours_back=2):
    """
    Fetch latest SPOT-1644 embedded history via Sofar public download API.
    Returns dict with buoy_Hs, buoy_Tp, buoy_Dp, buoy_time or None on failure.
    """
    import requests
    import io
    from datetime import datetime, timezone, timedelta

    TOKEN = "1bc9848d3e524c34a1eb220e121d9a9e"
    end   = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours_back)

    try:
        resp = requests.get(
            "https://api.sofarocean.com/fetch/download",
            params={
                "spotterId":         "SPOT-1644",
                "startDate":         start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "endDate":           end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "processingSources": "all",
            },
            headers={
                "View_token": TOKEN,
                "Origin":     "https://spotter.sofarocean.com",
                "Referer":    "https://spotter.sofarocean.com/",
            },
            timeout=15,
        )
        resp.raise_for_status()

    except requests.exceptions.Timeout:
        print("WARN: SPOT fetch timed out — skipping buoy obs", file=sys.stderr)
        return None
    except requests.exceptions.HTTPError as e:
        print(f"WARN: SPOT fetch HTTP error {e} — skipping buoy obs", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"WARN: SPOT fetch failed ({e}) — skipping buoy obs", file=sys.stderr)
        return None

    try:
        df = pd.read_csv(io.StringIO(resp.text))
        # filter to rows with valid Hs
        df = df[df['Significant Wave Height (m)'] != '-'].copy()
        if df.empty:
            print("WARN: SPOT response had no valid wave obs — skipping", file=sys.stderr)
            return None

        df['ts'] = pd.to_datetime(df['Epoch Time'].astype(int), unit='s', utc=True)
        df['Hs'] = df['Significant Wave Height (m)'].astype(float)
        df['Tp'] = df['Peak Period (s)'].astype(float)
        df['Dp'] = df['Peak Direction (deg)'].astype(float)
        df = df.sort_values('ts')

        latest = df.iloc[-1]
        print(f"SPOT loaded: Hs={latest['Hs']:.2f}m Tp={latest['Tp']:.1f}s Dp={latest['Dp']:.0f}° at {latest['ts']}")
        return {
            'buoy_Hs':   round(float(latest['Hs']), 3),
            'buoy_Tp':   round(float(latest['Tp']), 2),
            'buoy_Dp':   round(float(latest['Dp']), 1),
            'buoy_time': latest['ts'].isoformat(),
        }

    except Exception as e:
        print(f"WARN: SPOT parse error ({e}) — skipping buoy obs", file=sys.stderr)
        return None

# --- Pull latest MOP nowcast ---
try:
    NOWCAST_URL = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_alongshore/B0385_nowcast.nc"
    ds_mop  = xr.open_dataset(NOWCAST_URL, engine='pydap')
    times   = ds_mop['waveTime'].values
    Hs_vals = ds_mop['waveHs'].values

    # Get most recent valid value
    valid   = (Hs_vals > 0) & (Hs_vals < 20)
    idx     = np.where(valid)[0][-1]
    Hs_mop  = float(Hs_vals[idx])
    mop_time = pd.Timestamp(times[idx], tz='UTC')
    print(f'MOP loaded: {Hs_mop:.2f}m at {mop_time}')

except Exception as e:
    print(f'ERROR loading MOP: {e}', file=sys.stderr)
    sys.exit(1)

# --- Pull latest Harvest ---
try:
    HARVEST_URL = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/071p1_rt.nc"
    ds_h    = xr.open_dataset(HARVEST_URL, engine='pydap')
    times_h = ds_h['waveTime'].values
    Hs_h    = ds_h['waveHs'].values
    Dp_h    = ds_h['waveDp'].values
    flag_h  = ds_h['waveFlagPrimary'].values

    valid_h     = (flag_h == 1) & (Hs_h > 0) & (Hs_h < 20)
    idx_h       = np.where(valid_h)[0][-1]
    Dp_harvest  = float(Dp_h[idx_h])
    Hs_harvest  = float(Hs_h[idx_h])
    harvest_time = pd.Timestamp(times_h[idx_h], tz='UTC')
    print(f'Harvest loaded: {Hs_harvest:.2f}m from {Dp_harvest:.0f}° at {harvest_time}')

except Exception as e:
    print(f'ERROR loading Harvest: {e}', file=sys.stderr)
    sys.exit(1)

# --- Fetch SPOT buoy observation ---
spot = fetch_spot_observation(hours_back=2)

# --- Load and refit wave model ---
wave_model = WaveModel()
wave_model.load(STATE_PATH)
train_df = load_training_data()
if len(train_df) >= 30:
    wave_model.fit(train_df)
    wave_model.save(STATE_PATH)

# --- Apply correction ---
result = correct_forecast(Hs_mop, Dp_harvest, model)
Hs_ml = wave_model.predict(
    mop_raw=Hs_mop,
    harvest_Hs=Hs_harvest,
    harvest_Dp=Dp_harvest,
    harvest_Tp=None,  # add once live log has Tp
)
print(f"ML prediction: {Hs_ml}m")

# --- Build output ---
output = {
    'generated_at': pd.Timestamp.now(tz='UTC').isoformat(),
    'current': {
        'time':         mop_time.isoformat(),
        'Hs_mop_raw':   round(Hs_mop, 3),
        'Hs_corrected': result['Hs_corrected'],
        'confidence':   result['confidence'],
        'bin':          result['bin'],
        'scale_factor': result['scale_factor'],
        'harvest': {
            'time': harvest_time.isoformat(),
            'Hs':   round(Hs_harvest, 3),
            'Dp':   round(Dp_harvest, 1),
        }
    }
}

# --- Write JSON ---
output_path = 'forecast_output.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
    
# --- Append to verification log ---
log_path = 'forecast_log.csv'
log_row = {
    'generated_at':  pd.Timestamp.now(tz='UTC').isoformat(),
    'mop_time':      mop_time.isoformat(),
    'Hs_mop_raw':    round(Hs_mop, 3),
    'Hs_corrected':  result['Hs_corrected'],
    'confidence':    result['confidence'],
    'bin':           result['bin'],
    'scale_factor':  round(result['scale_factor'], 4),
    'harvest_time':  harvest_time.isoformat(),
    'harvest_Hs':    round(Hs_harvest, 3),
    'harvest_Dp':    round(Dp_harvest, 1),
    'buoy_Hs':   spot['buoy_Hs']   if spot else None,
    'buoy_Tp':   spot['buoy_Tp']   if spot else None,
    'buoy_Dp':   spot['buoy_Dp']   if spot else None,
    'buoy_time': spot['buoy_time'] if spot else None,
    'Hs_ml': round(Hs_ml, 3) if Hs_ml is not None else None,
}

log_df = pd.DataFrame([log_row])
write_header = not pd.io.common.file_exists(log_path)
log_df.to_csv(log_path, mode='a', header=write_header, index=False)
print(f'Appended to {log_path}')

print(f'\nCampus Point Wave Forecast')
print(f'Generated: {pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")}')
print(f'MOP raw:      {Hs_mop:.2f}m  ({mop_time.strftime("%Y-%m-%d %H:%M UTC")})')
print(f'Harvest:      {Hs_harvest:.2f}m from {Dp_harvest:.0f}°')
print(f'Swell bin:    {result["bin"]}')
print(f'Hs corrected: {result["Hs_corrected"]}m')
print(f'Confidence:   {result["confidence"]}')
print(f'Written to {output_path}')