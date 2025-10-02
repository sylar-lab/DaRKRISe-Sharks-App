import streamlit as st
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="Interactive Map", layout="wide")
st.title("Sharks Foraging Detection: Interactive Map")

st.markdown("""
This interactive map allows you to zoom, pan, and explore the world. In the future, you will be able to mark possible foraging zones and other areas of interest.
""")



import numpy as np
import random




# --- Simulate ocean productivity (chlorophyll-a proxy) ---
@st.cache_data(show_spinner=False)
def simulate_productivity_grid(n_points=40):
	lats = np.random.uniform(-60, 60, n_points)
	lons = np.random.uniform(-180, 180, n_points)
	# Simulate productivity: higher near equator, but less saturated
	productivity = np.clip(0.7 * np.cos(np.radians(lats)) + 0.1 * np.random.randn(n_points), 0, None)
	return list(zip(lats, lons, productivity))

# --- Simulate shark locations ---
@st.cache_data(show_spinner=False)
def simulate_sharks(n_sharks=10):
	sharks = []
	for _ in range(n_sharks):
		lat = random.uniform(-50, 50)
		lon = random.uniform(-170, 170)
		sharks.append((lat, lon))
	return sharks

# --- Refresh Button ---
col1, col2 = st.columns([1, 8])
with col1:
	if st.button('Refresh Map', help='Regenerate shark locations and productivity'):
		st.cache_data.clear()



# --- Create map with blue sea style ---
m = folium.Map(
	location=[0, 0],
	zoom_start=2,
	control_scale=True,
	tiles='https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
	attr='Tiles &copy; Esri &mdash; Source: Esri, GEBCO, NOAA, National Geographic, DeLorme, NAVTEQ, and other contributors',
	max_zoom=10
)

# Add productivity heatmap with softer look
from folium.plugins import HeatMap
productivity_points = simulate_productivity_grid()
heat_data = [[lat, lon, prod] for lat, lon, prod in productivity_points]
HeatMap(
	heat_data,
	min_opacity=0.08,
	max_opacity=0.25,
	radius=16,
	blur=18,
	gradient={0.2: '#225ea8', 0.5: '#41b6c4', 0.8: '#a1dab4', 1: '#ffffcc'},
	name='Productivity'
).add_to(m)

# Add PNG shark markers
shark_locs = simulate_sharks()
# PNG shark icon (public domain):
shark_icon_url = 'https://cdn-icons-png.flaticon.com/512/616/616408.png'
for idx, (lat, lon) in enumerate(shark_locs):
	folium.Marker(
		location=[lat, lon],
		icon=folium.CustomIcon(shark_icon_url, icon_size=(32, 32)),
		popup=f"Shark #{idx+1}",
		tooltip=f"Shark #{idx+1}"
	).add_to(m)

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
	width=1800,
	height=1200,
	center=center,
	zoom=zoom,
	feature_group_to_add=feature_group
)

# --- Conceptual Tag Model ---
with st.sidebar:
	st.header("Conceptual Shark Tag Model")
	st.markdown("""
	**Features:**
	- Real-time GPS location tracking
	- Depth and temperature sensors
	- Stomach content analysis (e.g., via pH, DNA, or micro-camera)
	- Accelerometer for behavior/activity
	- Real-time satellite uplink for data transmission
	- Long battery life, hydrodynamic design
    
	**How it works:**
	1. Tag attaches to shark's dorsal fin.
	2. Continuously records location, depth, and feeding events.
	3. When shark surfaces, tag transmits data to satellite.
	4. Data is used to predict foraging zones and understand diet in near real-time.
	""")

st.info("Shark locations (red markers) and simulated ocean productivity (heatmap) are shown. Foraging zones can be predicted where productivity is high and sharks are present.")

