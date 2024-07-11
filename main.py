import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from geomdl import NURBS
from geomdl import utilities as utils
import plotly.graph_objects as go

# Function to generate points with fixed spacing along a curve
def generate_points_with_fixed_spacing(x_curve, y_curve, spacing=0.05):
    points = [(x_curve[0], y_curve[0])]
    accumulated_distance = 0.0
    
    for i in range(1, len(x_curve)):
        dx = x_curve[i] - x_curve[i-1]
        dy = y_curve[i] - y_curve[i-1]
        segment_length = np.sqrt(dx**2 + dy**2)
        accumulated_distance += segment_length
        
        while accumulated_distance >= spacing:
            ratio = (spacing - (accumulated_distance - segment_length)) / segment_length
            new_x = x_curve[i-1] + ratio * dx
            new_y = y_curve[i-1] + ratio * dy
            points.append((new_x, new_y))
            accumulated_distance -= spacing

    return np.array(points)

# Function to fit a smooth NURBS curve and generate points with fixed spacing
def fit_smooth_nurbs_curve(x_curve, y_curve, degree=3, sample_size=100):
    # Create a NURBS curve instance
    curve = NURBS.Curve()

    # Set the degree of the curve
    curve.degree = degree

    # Set control points
    curve.ctrlpts = np.array(list(zip(x_curve, y_curve))).tolist()

    # Generate a uniform knot vector
    curve.knotvector = utils.generate_knot_vector(curve.degree, len(curve.ctrlpts))

    # Set sample size
    curve.sample_size = sample_size

    # Evaluate curve points
    curve.evaluate()

    return np.array(curve.evalpts)

# Load the data (assuming Data.csv contains x, y, z columns)
df1 = pd.read_csv('Data.csv', sep=';')
df2 = pd.read_csv('76er_Default Dataset-12.csv', sep=';')
df3 = pd.read_csv('88,5er_Default Dataset-12.csv', sep=';')
df4 = pd.read_csv('Data087.csv', sep=';')

df1.columns = ['x', 'y', 'z']
df2.columns = ['x', 'y']
df3.columns = ['x', 'y']
df4.columns = ['x', 'y', 'z']
# Filter out z-values 0.885 and 0.76
z = df1['z'].values
mask =~np.isclose(z, 0.885) & ~np.isclose(z, 0.76) & ~np.isclose(z, 0.87)
df1 = df1[mask]


# Add a constant value column to df2
df2['z'] = 0.76

# Add a constant value column to df3
df3['z'] = 0.885

# Concatenate the DataFrames
df_combined = pd.concat([df1, df2, df3,df4])

# Extract x, y, and z values from the dataframe
x = df_combined['x'].values
y = df_combined['y'].values
z = df_combined['z'].values

# Filter out z-values 0.78 and 0.8
mask = ~np.isclose(z, 0.78) & ~np.isclose(z, 0.8)
x_filtered = x[mask]
y_filtered = y[mask]
z_filtered = z[mask]

# Normalize x and y
x_min, x_max = np.min(x_filtered), np.max(x_filtered)
y_min, y_max = np.min(y_filtered), np.max(y_filtered)

x_normalized = (x_filtered - x_min) / (x_max - x_min)
y_normalized = (y_filtered - y_min) / (y_max - y_min)

# Get unique z-values excluding the filtered ones
unique_z = np.unique(z_filtered)

# Collect points for NURBS
nurbs_points = []

# Fit NURBS curves and collect points
for z_value in unique_z:
    # Filter data for current z-value
    indices = np.where(z_filtered == z_value)[0]
    x_curve = x_normalized[indices]
    y_curve = y_normalized[indices]

    # Fit a smooth NURBS curve and generate points
    smooth_nurbs_points = fit_smooth_nurbs_curve(x_curve, y_curve, degree=5, sample_size=500)
    
    # Generate points with fixed spacing along the smooth curve
    fixed_spacing_points = generate_points_with_fixed_spacing(smooth_nurbs_points[:, 0], smooth_nurbs_points[:, 1], spacing=0.05)
    
    # Collect points with the corresponding z-value
    for pt in fixed_spacing_points:
        # Denormalize the points
        pt_original = [(pt[0] * (x_max - x_min) + x_min), (pt[1] * (y_max - y_min) + y_min)]
        nurbs_points.append([pt_original[0], pt_original[1], z_value])

# Convert nurbs points to numpy array
nurbs_points = np.array(nurbs_points)

# Extract x, y, and z values from nurbs points
x_nurbs = nurbs_points[:, 0]
y_nurbs = nurbs_points[:, 1]
z_nurbs = nurbs_points[:, 2]

# Normalize x_nurbs and y_nurbs
x_nurbs_normalized = (x_nurbs - x_min) / (x_max - x_min)
y_nurbs_normalized = (y_nurbs - y_min) / (y_max - y_min)

# Create a high-resolution grid of x, y points
grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 200))

# Interpolate z-values for the grid points using griddata with cubic method
grid_z = griddata((x_nurbs_normalized, y_nurbs_normalized), z_nurbs, (grid_x, grid_y), method='cubic')

# Handle NaN values in grid_z
#grid_z = np.nan_to_num(grid_z, nan=0.0, posinf=0.0, neginf=0.0)

# Denormalize the grid coordinates for plotting
grid_x_denormalized = grid_x * (x_max - x_min) + x_min
grid_y_denormalized = grid_y * (y_max - y_min) + y_min

# Streamlit App
st.title('Muscheldiagramm')

# User input for coordinates
st.sidebar.header('Input')
coordinate_input = st.sidebar.text_area("Enter x, y coordinates (comma-separated, one pair per line)", "25, 600\n26, 600\n27,600\n21,575")
coordinate_list = [tuple(map(float, line.split(','))) for line in coordinate_input.split('\n') if line]

# Convert coordinates to DataFrame for display and interpolation
input_df = pd.DataFrame(coordinate_list, columns=['x', 'y'])

# Interpolate z-values for input coordinates
if not input_df.empty:
    input_x_normalized = (input_df['x'].values - x_min) / (x_max - x_min)
    input_y_normalized = (input_df['y'].values - y_min) / (y_max - y_min)
    input_z = griddata((x_nurbs_normalized, y_nurbs_normalized), z_nurbs, (input_x_normalized, input_y_normalized), method='cubic')
    input_df['z'] = input_z

    # Display the table underneath the input box
    st.sidebar.subheader("Output")
    st.sidebar.write(input_df)

# Sort grid_z for plotting contours

# Introduce NaN values in grid_z (example)
#grid_z[grid_z > 2] = np.nan

# Handle NaN values in grid_z
masked_z = np.ma.masked_invalid(grid_z)
z_min = np.nanmin(masked_z)
z_max = np.nanmax(masked_z)


# Flatten arrays for sorting
flat_indices = np.arange(grid_z.size)
sorted_indices = np.argsort(grid_z.flatten())
sorted_grid_z = grid_z.flatten()[sorted_indices]
sorted_grid_x = grid_x_denormalized.flatten()[sorted_indices]
sorted_grid_y = grid_y_denormalized.flatten()[sorted_indices]

# Create Plotly figure with contour plot
fig = go.Figure(go.Contour(
    z=sorted_grid_z,
    x=sorted_grid_x,
    y=sorted_grid_y,
    colorscale='Viridis',  # Adjust colorscale as needed
    colorbar=dict(title='Z-values'),  # Add color bar
    contours=dict(
        start=np.nanmin(grid_z),  # Minimum value of z
        end=np.nanmax(grid_z),    # Maximum value of z
        size=0.005,               # Contour line interval
        showlabels=True,          # Show labels on contour lines
        labelfont=dict(
            size=12,
            color='white',
        ),
    ),
    hoverinfo='skip'
))

# Plot the NURBS curves for each unique z value
for z_value in unique_z:
    indices = np.where(z_nurbs == z_value)
    fig.add_trace(go.Scatter(
        x=x_nurbs[indices],
        y=y_nurbs[indices],
        mode='lines',
        line=dict(dash='dash'),
        name=f'z={z_value:.3f}',
        hoverinfo='skip',
        visible='legendonly',  # Initially hidden but shown in legend
    ))

# Plot input coordinates with hover only for input points
if not input_df.empty:
    input_df['hover_text'] = "Point "+ input_df.index.astype(str) + "<br>z=" + input_df['z'].round(3).astype(str)
    fig.add_trace(go.Scatter(
        x=input_df['x'],
        y=input_df['y'],
        mode='markers',
        marker=dict(color='red', size=10),
        name="Input",
        text=input_df['hover_text'], 
        hoverinfo='text',    # Show hover information as text
    ))

# Update layout
fig.update_layout(
    title="Nurbs und Kontur Diagramm",
    xaxis=dict(title='X'),
    yaxis=dict(title='Y'),
    legend=dict(traceorder='normal', orientation='h'),
    autosize=False,
    width=800,
    height=600,
)

# Show plot using Streamlit
st.plotly_chart(fig, use_container_width=True)
