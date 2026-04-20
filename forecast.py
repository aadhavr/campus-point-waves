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

def fetch_tide(hours_back=1):
    """
    Fetch latest water level + next 6h tide predictions from NOAA.
    Station 9411340 = Santa Barbara.
    Returns dict with tide_height_m, tide_trend, next_high, next_low or None on failure.
    """
    import requests
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    # water level: last hour actual
    # predictions: next 6 hours
    base = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    try:
        # current water level
        wl_resp = requests.get(base, params={
            "station":     "9411340",
            "product":     "water_level",
            "datum":       "MLLW",
            "time_zone":   "gmt",
            "units":       "metric",
            "format":      "json",
            "begin_date":  (now - timedelta(hours=1)).strftime("%Y%m%d %H:%M"),
            "end_date":    now.strftime("%Y%m%d %H:%M"),
        }, timeout=10)
        wl_resp.raise_for_status()
        wl_data = wl_resp.json()
        readings = wl_data.get("data", [])
        if not readings:
            print("WARN: NOAA water level returned no data", file=sys.stderr)
            return None
        latest_wl = float(readings[-1]["v"])
        prev_wl   = float(readings[0]["v"])
        trend     = "rising" if latest_wl > prev_wl else "falling"

        # next high/low predictions
        pred_resp = requests.get(base, params={
            "station":     "9411340",
            "product":     "predictions",
            "datum":       "MLLW",
            "time_zone":   "gmt",
            "units":       "metric",
            "format":      "json",
            "interval":    "hilo",
            "begin_date":  now.strftime("%Y%m%d %H:%M"),
            "end_date":    (now + timedelta(hours=24)).strftime("%Y%m%d %H:%M"),
        }, timeout=10)
        pred_resp.raise_for_status()
        pred_data = pred_resp.json()
        preds = pred_data.get("predictions", [])

        next_high = next((p for p in preds if p["type"] == "H"), None)
        next_low  = next((p for p in preds if p["type"] == "L"), None)

        result = {
            "tide_height_m": round(latest_wl, 2),
            "tide_trend":    trend,
            "next_high_m":   round(float(next_high["v"]), 2) if next_high else None,
            "next_high_t":   next_high["t"] if next_high else None,
            "next_low_m":    round(float(next_low["v"]), 2)  if next_low  else None,
            "next_low_t":    next_low["t"]  if next_low  else None,
        }
        print(f"Tide loaded: {latest_wl:.2f}m MLLW ({trend}), "
              f"next high {result['next_high_m']}m @ {result['next_high_t']}, "
              f"next low {result['next_low_m']}m @ {result['next_low_t']}")
        return result

    except Exception as e:
        print(f"WARN: tide fetch failed ({e}) — skipping", file=sys.stderr)
        return None


def fetch_wind(lat=34.4140, lon=-119.8489):
    """
    Fetch current wind at Campus Point from Open-Meteo (free, no key).
    lat/lon default = Campus Point, UCSB.
    Returns dict with wind_speed_ms, wind_dir_deg, wind_gust_ms or None on failure.
    """
    import requests

    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":        lat,
                "longitude":       lon,
                "current":         "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
                "wind_speed_unit": "ms",
                "timezone":        "UTC",
                "forecast_days":   1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data    = resp.json()
        current = data["current"]

        speed   = current["wind_speed_10m"]
        direction = current["wind_direction_10m"]
        gust    = current["wind_gusts_10m"]

        # classify wind relative to Campus Point
        # offshore = easterly (45–135°), onshore = westerly (225–315°)
        if 45 <= direction <= 135:
            wind_class = "offshore"
        elif 225 <= direction <= 315:
            wind_class = "onshore"
        elif direction < 45 or direction > 315:
            wind_class = "sideshore-N"
        else:
            wind_class = "sideshore-S"

        result = {
            "wind_speed_ms":  round(speed, 1),
            "wind_dir_deg":   round(direction, 1),
            "wind_gust_ms":   round(gust, 1),
            "wind_class":     wind_class,
        }
        print(f"Wind loaded: {speed:.1f}m/s from {direction:.0f}° ({wind_class}), gusts {gust:.1f}m/s")
        return result

    except Exception as e:
        print(f"WARN: wind fetch failed ({e}) — skipping", file=sys.stderr)
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

# --- Fetch tide and wind ---
tide = fetch_tide()
wind = fetch_wind()

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
        },
        'tide': tide,
        'wind': wind,
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
    # tide
    'tide_height_m': tide['tide_height_m'] if tide else None,
    'tide_trend':    tide['tide_trend']    if tide else None,
    'next_high_m':   tide['next_high_m']   if tide else None,
    'next_high_t':   tide['next_high_t']   if tide else None,
    'next_low_m':    tide['next_low_m']    if tide else None,
    'next_low_t':    tide['next_low_t']    if tide else None,
    # wind
    'wind_speed_ms': wind['wind_speed_ms'] if wind else None,
    'wind_dir_deg':  wind['wind_dir_deg']  if wind else None,
    'wind_gust_ms':  wind['wind_gust_ms']  if wind else None,
    'wind_class':    wind['wind_class']    if wind else None,
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
if tide:
    print(f'Tide:         {tide["tide_height_m"]}m MLLW ({tide["tide_trend"]}) — '
          f'next high {tide["next_high_m"]}m @ {tide["next_high_t"]}, '
          f'next low {tide["next_low_m"]}m @ {tide["next_low_t"]}')
if wind:
    print(f'Wind:         {wind["wind_speed_ms"]}m/s from {wind["wind_dir_deg"]}° '
          f'({wind["wind_class"]}), gusts {wind["wind_gust_ms"]}m/s')
print(f'Written to {output_path}')