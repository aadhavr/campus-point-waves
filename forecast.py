# forecast.py
# Pulls latest MOP and Harvest data, applies bias correction,
# outputs a corrected wave forecast for Campus Point.
# Runs hourly via GitHub Actions.

import json
import sys
import numpy as np
import pandas as pd
import xarray as xr

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

# --- Apply correction ---
result = correct_forecast(Hs_mop, Dp_harvest, model)

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

print(f'\nCampus Point Wave Forecast')
print(f'Generated: {pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")}')
print(f'MOP raw:      {Hs_mop:.2f}m  ({mop_time.strftime("%Y-%m-%d %H:%M UTC")})')
print(f'Harvest:      {Hs_harvest:.2f}m from {Dp_harvest:.0f}°')
print(f'Swell bin:    {result["bin"]}')
print(f'Hs corrected: {result["Hs_corrected"]}m')
print(f'Confidence:   {result["confidence"]}')
print(f'Written to {output_path}')