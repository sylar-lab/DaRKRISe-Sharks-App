
import streamlit as st
import numpy as np
import folium
import pickle
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from sparse_lgcp import SparseLGCP

st.set_page_config(
	page_title="Home",
	layout="wide",
	initial_sidebar_state="expanded",
	menu_items={
		"About": "Sharks Foraging Detection Version 1.0",
		"Get help": "https://github.com/sylar-lab/DaRKRISe-Sharks-App",
		"Report a bug": "https://github.com/sylar-lab/DaRKRISe-Sharks-App/issues"
	}
)

# --- Session-state-based map/data caching ---
if 'productivity_points' not in st.session_state or 'shark_locs' not in st.session_state or 'pred_heat' not in st.session_state or 'last_refresh' not in st.session_state:
	st.session_state['productivity_points'] = None
	st.session_state['shark_locs'] = None
	st.session_state['pred_heat'] = None
	st.session_state['last_refresh'] = 0

# --- Sidebar user controls for model and map ---
with st.sidebar:
	st.header("Model & Map Controls")
	num_samples = st.slider("MC samples for prediction", min_value=100, max_value=2000, value=500, step=100)
	grid_lat_res = st.slider("Prediction grid latitude points", min_value=10, max_value=100, value=40, step=5)
	grid_lon_res = st.slider("Prediction grid longitude points", min_value=20, max_value=200, value=80, step=10)
	heatmap_opacity = st.slider("Heatmap max opacity", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
	heatmap_radius = st.slider("Heatmap radius", min_value=5, max_value=50, value=18, step=1)

col1, col2 = st.columns([1, 8])
with col1:
	refresh = st.button('Refresh Map', help='Regenerate shark locations and productivity')

if refresh or st.session_state['productivity_points'] is None:
	# Regenerate data
	lats = np.random.uniform(-60, 60, 40)
	lons = np.random.uniform(-180, 180, 40)
	productivity = np.clip(0.7 * np.cos(np.radians(lats)) + 0.1 * np.random.randn(40), 0, None)
	st.session_state['productivity_points'] = list(zip(lats, lons, productivity))
	# Load shark locations from dataset with error handling
	import pandas as pd
	import os
	ds_name = 'sharks_spatial_filtered.csv'
	ds_path = './data'
	try:
		df = pd.read_csv(os.path.join(ds_path, ds_name))
		if 'latitude' in df.columns and 'longitude' in df.columns:
			sharks = list(zip(df['latitude'], df['longitude']))
			if len(sharks) > 1000:
				st.warning(f"Dataset has {len(sharks)} shark locations. Showing only the first 1000 for performance.")
				sharks = sharks[:1000]
			st.session_state['shark_locs'] = sharks
		else:
			st.session_state['shark_locs'] = []
			st.warning("CSV file does not contain 'latitude' and 'longitude' columns.")
	except Exception as e:
		st.session_state['shark_locs'] = []
		st.warning(f"Could not load shark locations: {e}")
	# Model prediction
	try:
		with open("models/sparse_lgcp.pkl", "rb") as f:
			model = pickle.load(f)
		# Use the same bounding box as model training for normalization
		lat_min, lat_max = 8, 55
		lon_min, lon_max = -98, -25

		grid_lat = np.linspace(lat_min, lat_max, grid_lat_res)
		grid_lon = np.linspace(lon_min, lon_max, grid_lon_res)
		t_last = 1.0
		grid_points = np.array([[lat, lon, t_last] for lat in grid_lat for lon in grid_lon])
		# Normalize using the same logic as model training
		lat_norm = (grid_points[:, 0] - lat_min) / (lat_max - lat_min)
		lon_norm = (grid_points[:, 1] - lon_min) / (lon_max - lon_min)
		coords_pred = np.column_stack([lon_norm, lat_norm, np.full_like(lat_norm, t_last)])
		rate_mean, _, _ = model.predict_rate(coords_pred, num_samples=num_samples, alpha_regularization=True)
		pred_heat = [
			[lat, lon, float(rate)]
			for (lat, lon, _), rate in zip(grid_points, rate_mean)
		]
		st.session_state['pred_heat'] = pred_heat
		st.session_state['last_refresh'] += 1
		# Debug: show summary stats for prediction
		rates = [r[2] for r in pred_heat]
		st.info(f"Prediction stats: min={np.min(rates):.3g}, max={np.max(rates):.3g}, mean={np.mean(rates):.3g}, nonzero={np.count_nonzero(rates)} / {len(rates)}")
		st.success("Model prediction overlay added to the map.")
	except Exception as e:
		st.session_state['pred_heat'] = None
		st.warning(f"Model prediction not available: {e}")

# --- Create map with blue sea style ---
center = st.session_state.get('map_center', [0, 0])
if isinstance(center, dict):
	center = [center.get('lat', 0), center.get('lng', 0)]
m = folium.Map(
	location=center,
	zoom_start=st.session_state.get('map_zoom', 2),
	control_scale=True,
	tiles='https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
	attr='Tiles &copy; Esri &mdash; Source: Esri, GEBCO, NOAA, National Geographic, DeLorme, NAVTEQ, and other contributors',
	max_zoom=10,
	no_wrap=True
)

st.markdown("### Shark Foraging Prediction (Model Output)")
if st.session_state['pred_heat'] is not None:
	HeatMap(
		st.session_state['pred_heat'],
		min_opacity=0.2,
		max_opacity=heatmap_opacity,
		radius=heatmap_radius,
		blur=22,
		gradient={
			0.0: '#ffffff',
			0.2: '#2c7bb6',
			0.4: '#abd9e9',
			0.6: '#ffffbf',
			0.8: '#fdae61',
			1.0: '#d7191c'
		},
		name=f'Model Prediction_{st.session_state["last_refresh"]}'
	).add_to(m)

# Add regular markers for the last 100 shark detections
if st.session_state['shark_locs']:
	last_sharks = st.session_state['shark_locs'][-100:]
	for idx, (lat, lon) in enumerate(last_sharks, 1):
		folium.Marker(
			location=[lat, lon],
			popup=f"Shark #{idx} ({lat:.2f}, {lon:.2f})",
			tooltip=f"Shark #{idx}"
		).add_to(m)
else:
	st.warning("No shark locations to display on the map.")

# Display the map in Streamlit with better interactivity
center = [0, 0]
zoom = 2
feature_group = None
for fg in m._children.values():
	if isinstance(fg, folium.map.FeatureGroup):
		feature_group = fg
		break

st_folium(
	m,
	center=center,
	zoom=zoom,
	feature_group_to_add=feature_group,
	key=f"folium_map_{st.session_state['last_refresh']}",
	use_container_width=True
)


st.info("Shark locations (red markers) and simulated ocean productivity (heatmap) are shown. Foraging zones can be predicted where productivity is high and sharks are present.")

