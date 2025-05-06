import streamlit as st
import math
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Configuración de la Página de Streamlit ---
st.set_page_config(layout="wide", page_title="Simulador de Proyectiles Animado")

st.title("Simulador de Proyectiles Interactivo y Animado")

# --- Constantes Físicas ---
G = 9.81 # Aceleración de la gravedad (m/s²)

# --- Variables de Estado Inicial y Parámetros ---
# Usamos st.session_state para mantener el estado entre reruns
if 'simulating' not in st.session_state:
    st.session_state.simulating = False
if 'history_time' not in st.session_state:
    st.session_state.history_time = []
if 'history_x_pos' not in st.session_state:
    st.session_state.history_x_pos = []
if 'history_y_pos' not in st.session_state:
    st.session_state.history_y_pos = []
if 'history_vx' not in st.session_state:
    st.session_state.history_vx = []
if 'history_vy' not in st.session_state:
    st.session_state.history_vy = []
if 'history_ax' not in st.session_state:
    st.session_state.history_ax = []
if 'history_ay' not in st.session_state:
    st.session_state.history_ay = []
if 'history_f_net_mag' not in st.session_state:
    st.session_state.history_f_net_mag = []
if 'projectile_pos_m' not in st.session_state:
    st.session_state.projectile_pos_m = [0.0, 0.0]
if 'projectile_vel_m' not in st.session_state:
    st.session_state.projectile_vel_m = [0.0, 0.0]
if 'time_elapsed' not in st.session_state:
     st.session_state.time_elapsed = 0.0


# --- Sidebar para Parámetros de Entrada ---
st.sidebar.header("Parámetros de Simulación")

initial_velocity = st.sidebar.slider("Velocidad Inicial (m/s)", 0.0, 100.0, 50.0, 0.5)
angle_degrees = st.sidebar.slider("Ángulo (°)", 0.0, 90.0, 45.0, 0.5)
initial_height_m = st.sidebar.slider("Altura Inicial (m)", 0.0, 50.0, 0.0, 0.1)
mass_kg = st.sidebar.slider("Masa (kg)", 0.1, 10.0, 1.0, 0.1)
# Sliders para la escala de los vectores en la gráfica
# Ajusta el rango de 0.01 a 1.0 para dar más control sobre la escala
force_vector_scale = st.sidebar.slider("Escala Vector Fuerza Neta", 0.01, 1.0, 0.05, 0.01)
velocity_vector_scale = st.sidebar.slider("Escala Vector Velocidad", 0.01, 1.0, 0.1, 0.01)

# Slider para controlar la velocidad de la animación
animation_speed_factor = st.sidebar.slider("Factor de Velocidad Animación", 0.1, 5.0, 1.0, 0.1) # 1.0 es velocidad normal, 2.0 es doble velocidad, etc.


# --- Funciones Auxiliares y de Cálculo ---

def calculate_predictions(v0, angle_deg, y0, G_val):
    """Calcula predicciones y retorna valores clave."""
    angle_rad = math.radians(angle_deg)
    v0x = v0 * math.cos(angle_rad)
    v0y = v0 * math.sin(angle_rad)

    if G_val <= 1e-9: # Treat very small or non-positive G as zero
         arbitrary_large_time = 100.0
         time_to_peak = arbitrary_large_time if v0y > 1e-9 else 0.0 # Peak time if moving up
         max_height_m = y0 + v0y * arbitrary_large_time if v0y > 1e-9 else y0 # Max height if moving up, else initial
         total_time = arbitrary_large_time if v0 > 1e-9 else 0.0 # Fly forever or just stay
         max_range_m = v0x * total_time

    else: # G > 0
        # Time to peak depends only on initial_vel_y and G
        time_to_peak = v0y / G_val if v0y > 1e-9 else 0.0 # Only has a peak if initial vertical velocity is positive
        # Max height depends on initial_vel_y, G, and initial_height
        max_height_m_rel = (v0y**2) / (2 * G_val) if v0y > 1e-9 else 0.0 # Only get height boost if v0y > 0
        max_height_m = y0 + max_height_m_rel

        # Total time is when y = 0. Solve quadratic equation: 0 = y0 + v0y*t - 0.5*G*t^2
        # 0.5*G*t^2 - v0y*t - y0 = 0
        # a = 0.5*G, b = -v0y, c = -y0
        discriminant = (-v0y)**2 - 4 * (0.5 * G_val) * (-y0)
        discriminant = v0y**2 + 2 * G_val * y0

        if discriminant < -1e-9: # Allow small floating point errors
             total_time = 0.0
             max_range_m = 0.0
        else:
             sqrt_discriminant = math.sqrt(max(0, discriminant)) # Ensure sqrt is non-negative
             # Quadratic formula solutions: t = [-b ± sqrt(discriminant)] / 2a
             # a = 0.5*G_val, 2a = G_val, b = -v0y
             t_sol1 = (v0y + sqrt_discriminant) / G_val
             t_sol2 = (v0y - sqrt_discriminant) / G_val

             # Take the positive time that makes sense physically.
             total_time = max(t_sol1, t_sol2, 0.0)

             # Handle cases where total_time is excessively large but velocity is small or angle is 0
             if total_time > 1000 and v0 > 1e-9 and angle_degrees != 90:
                  total_time = 1000.0
             elif total_time > 1000 and angle_degrees == 90 and v0y > 0:
                   total_time = (v0y/G_val)*2 + 5


        max_range_m = v0x * total_time if total_time > 1e-9 and v0x >= -1e-9 else 0.0


    return {
        "v0x": v0x,
        "v0y": v0y,
        "time_to_peak": time_to_peak,
        "max_height_m": max_height_m,
        "total_time": total_time,
        "max_range_m": max_range_m
    }


def run_full_simulation(v0x, v0y, y0, mass, G_val, total_sim_time=None, num_steps=300): # Default steps increased
    """
    Runs the simulation from t=0 up to a specified time or landing.
    Returns data at a fixed number of steps for animation.
    """
    # If total_sim_time is not provided, calculate the landing time
    if total_sim_time is None:
         preds = calculate_predictions(math.sqrt(v0x**2 + v0y**2), math.degrees(math.atan2(v0y, v0x)), y0, G_val)
         total_sim_time = preds["total_time"]

    # Ensure total_sim_time is positive for steps, add a small epsilon if 0 but velocity > 0
    if total_sim_time <= 1e-9:
         if math.sqrt(v0x**2 + v0y**2) > 1e-9 or y0 > 1e-9: # If there's initial velocity or height
              total_sim_time = 1.0 # Simulate for a short duration
         else:
              total_sim_time = 0.0 # No movement, time is 0
              num_steps = 1 # Just one step for initial state

    # Generate time steps
    if num_steps > 1:
        time_hist = np.linspace(0, total_sim_time, num_steps).tolist()
    elif num_steps == 1 and total_sim_time >= 0:
         time_hist = [total_sim_time]
         if total_sim_time == 0: time_hist = [0.0]
    else:
         time_hist = [0.0]
         num_steps = 1

    # Calculate state at each time step using kinematic equations
    x_hist = [v0x * t for t in time_hist]
    y_hist = [y0 + v0y * t - 0.5 * G_val * t**2 for t in time_hist]
    vx_hist = [v0x for _ in time_hist]
    vy_hist = [v0y - G_val * t for t in time_hist]
    ax_hist = [0.0 for _ in time_hist]
    ay_hist = [-G_val if G_val > 1e-9 else 0.0 for _ in time_hist]
    fnet_hist = [mass * G_val if G_val > 1e-9 else 0.0 for _ in time_hist]

    # Ensure the final y position doesn't go drastically below ground if G>0
    if G_val > 1e-9 and total_sim_time > 1e-9:
         landing_idx = next((i for i, y in enumerate(y_hist[1:], start=1) if y < -0.01), len(y_hist))

         if landing_idx < len(y_hist) :
              time_hist = time_hist[:landing_idx + 1]
              x_hist = x_hist[:landing_idx + 1]
              y_hist = y_hist[:landing_idx + 1]
              vx_hist = vx_hist[:landing_idx + 1]
              vy_hist = vy_hist[:landing_idx + 1]
              ax_hist = ax_hist[:landing_idx + 1]
              ay_hist = ay_hist[:landing_idx + 1]
              fnet_hist = fnet_hist[:landing_idx + 1]

              if y_hist and y_hist[-1] < 0:
                   y_hist[-1] = 0.0


    return {
        "time": time_hist,
        "x_pos": x_hist,
        "y_pos": y_hist,
        "vx": vx_hist,
        "vy": vy_hist,
        "ax": ax_hist,
        "ay": ay_hist,
        "f_net_mag": fnet_hist,
        "final_pos": [x_hist[-1] if x_hist else 0.0, y_hist[-1] if y_hist else y0],
        "final_vel": [vx_hist[-1] if vx_hist else v0x, vy_hist[-1] if vy_hist else v0y],
        "final_time": time_hist[-1] if time_hist else 0.0
    }


def reset_simulation_state():
     """Resets the Streamlit session state for a new simulation."""
     st.session_state.simulating = False
     st.session_state.history_time = []
     st.session_state.history_x_pos = []
     st.session_state.history_y_pos = []
     st.session_state.history_vx = []
     st.session_state.history_vy = []
     st.session_state.history_ax = []
     st.session_state.history_ay = []
     st.session_state.history_f_net_mag = []
     st.session_state.projectile_pos_m = [0.0, 0.0]
     st.session_state.projectile_vel_m = [0.0, 0.0]
     st.session_state.time_elapsed = 0.0


# --- Botones de Control ---
col1, col2 = st.columns(2)

with col1:
    if st.button("Lanzar / Recalcular Simulación"):
        preds = calculate_predictions(initial_velocity, angle_degrees, initial_height_m, G)
        sim_results = run_full_simulation(
            preds["v0x"],
            preds["v0y"],
            initial_height_m,
            mass_kg,
            G,
            total_sim_time=preds["total_time"],
            num_steps=300
            )

        st.session_state.simulating = True
        st.session_state.history_time = sim_results["time"]
        st.session_state.history_x_pos = sim_results["x_pos"]
        st.session_state.history_y_pos = sim_results["y_pos"]
        st.session_state.history_vx = sim_results["vx"]
        st.session_state.history_vy = sim_results["vy"]
        st.session_state.history_ax = sim_results["ax"]
        st.session_state.history_ay = sim_results["ay"]
        st.session_state.history_f_net_mag = sim_results["f_net_mag"]
        st.session_state.projectile_pos_m = sim_results["final_pos"]
        st.session_state.projectile_vel_m = sim_results["final_vel"]
        st.session_state.time_elapsed = sim_results["final_time"]


with col2:
    if st.button("Reset"):
        reset_simulation_state()
        st.rerun()

# --- Mostrar Parámetros y Predicciones ---
st.subheader("Parámetros y Predicciones")

preds = calculate_predictions(initial_velocity, angle_degrees, initial_height_m, G)

param_col1, param_col2, param_col3 = st.columns(3)

with param_col1:
    st.write(f"**Velocidad Inicial (v₀):** {initial_velocity:.1f} m/s")
    st.write(f"**Ángulo (θ):** {angle_degrees:.1f}°")
    st.write(f"**Altura Inicial (y₀):** {initial_height_m:.1f} m")
    st.write(f"**Masa (m):** {mass_kg:.1f} kg")
    st.write(f"**Gravedad (g):** {G:.2f} m/s²")

with param_col2:
    st.write("**Predicciones (ignorando resistencia del aire):**")
    range_str = f"{preds['max_range_m']:.1f} m" if not math.isnan(preds['max_range_m']) and not math.isinf(preds['max_range_m']) and preds['max_range_m'] >= 0 else "N/A"
    time_str = f"{preds['total_time']:.2f} s" if not math.isnan(preds['total_time']) and not math.isinf(preds['total_time']) and preds['total_time'] >= 0 else "N/A"
    peak_t_str = f"{preds['time_to_peak']:.2f} s" if not math.isinf(preds['time_to_peak']) and preds['time_to_peak'] >= 0 else "N/A"

    st.write(f"**Alcance Máximo (X):** {range_str}")
    st.write(f"**Altura Máxima (Y):** {preds['max_height_m']:.1f} m")
    st.write(f"**Tiempo de Vuelo:** {time_str}")
    st.write(f"**Tiempo al Pico:** {peak_t_str}")
    st.write(f"**Velocidad Inicial X (v₀ₓ):** {preds['v0x']:.1f} m/s")
    st.write(f"**Velocidad Inicial Y (v₀ᵧ):** {preds['v0y']:.1f} m/s")


with param_col3:
     if st.session_state.simulating or len(st.session_state.history_time) > 0:
         st.write("**Estado Final de Simulación:**")
         final_time = st.session_state.time_elapsed
         final_pos = st.session_state.projectile_pos_m
         final_vel = st.session_state.projectile_vel_m
         final_f_net = mass_kg * G if G > 1e-9 else 0.0


         display_y = final_pos[1]
         display_x = final_pos[0]

         if G > 1e-9 and initial_height_m >= 0 and abs(final_pos[1]) < 0.1:
             display_y = 0.0


         st.write(f"**Tiempo:** {final_time:.2f} s")
         st.write(f"**Posición Final (x, y):** ({display_x:.1f} m, {display_y:.1f} m)")
         st.write(f"**Velocidad Final (vₓ, vᵧ):** ({final_vel[0]:.1f} m/s, {final_vel[1]:.1f} m/s)")
         st.write(f"**Magnitud Fuerza Neta:** {(mass_kg * G) if G > 1e-9 else 0.0:.2f} N")


# --- Visualización de la Simulación Animada (Trayectoria, Proyectil y Vectores) ---
st.subheader("Visualización de la Simulación Animada")

if st.session_state.simulating or len(st.session_state.history_time) > 0:
    history_time = st.session_state.history_time
    history_x = st.session_state.history_x_pos
    history_y = st.session_state.history_y_pos
    history_vx = st.session_state.history_vx
    history_vy = st.session_state.history_vy

    # Plotly figure
    fig_sim = go.Figure()

    # --- Base Traces ---
    # Add the TRACE THAT WILL BE ANIMATED to show the path covered so far
    # Initialized with only the first point(s)
    if history_x and history_y:
         # Traza de la trayectoria que se va dibujando (inicializada con el primer punto)
         fig_sim.add_trace(go.Scatter(x=[history_x[0]], y=[history_y[0]], mode='lines', name='Trayectoria Trazada',
                                      line=dict(color='green', width=2)))

         # Add the projectile marker trace (this data will be updated in frames)
         fig_sim.add_trace(go.Scatter(x=[history_x[0]], y=[history_y[0]], mode='markers', name='Proyectil',
                                      marker=dict(color='red', size=10)))

         # Add Velocity Vector trace (this data will be updated in frames)
         v0x = history_vx[0] if history_vx else 0
         v0y = history_vy[0] if history_vy else 0
         vel_end_x_init = history_x[0] + v0x * velocity_vector_scale
         vel_end_y_init = history_y[0] + v0y * velocity_vector_scale
         fig_sim.add_trace(go.Scatter(x=[history_x[0], vel_end_x_init], y=[history_y[0], vel_end_y_init],
                                      mode='lines', name='Velocidad', line=dict(color='blue', width=3)))

         # Add Force Vector trace (this data will be updated in frames)
         force_mag_init = mass_kg * G if G > 1e-9 else 0.0
         force_end_x_init = history_x[0]
         force_end_y_init = history_y[0] - force_mag_init * force_vector_scale
         # Changed color to RED and increased width
         fig_sim.add_trace(go.Scatter(x=[history_x[0], force_end_x_init], y=[history_y[0], force_end_y_init],
                                      mode='lines', name='Fuerza Neta', line=dict(color='red', width=3)))


         # --- Define Frames for Animation ---
         frames = []
         # Calculate duration per frame for real-time animation, adjusted by speed factor
         total_sim_duration = history_time[-1] if history_time else 0
         num_frames = len(history_time) if history_time else 1
         # Ensure calculation is valid and apply speed factor
         if total_sim_duration > 1e-9 and num_frames > 1 and animation_speed_factor > 0:
             # Duration for 1x speed / number of frames = duration per frame
             # Divide by animation_speed_factor to make it faster (factor > 1) or slower (factor < 1)
             frame_duration_ms = (total_sim_duration / num_frames) * 1000 / animation_speed_factor
             # Set a minimum duration to avoid issues with very fast animations
             frame_duration_ms = max(1, frame_duration_ms) # Minimum 1ms per frame
         else:
             # Default duration if calculation is invalid (e.g., duration 0, 1 frame, speed factor 0)
             frame_duration_ms = 0 if num_frames <= 1 else 50 # 0ms if only 1 frame, default 50ms otherwise


         for i in range(len(history_time)):
             current_x = history_x[i]
             current_y = history_y[i]
             current_vx = history_vx[i]
             current_vy = history_vy[i]
             current_force_mag = mass_kg * G if G > 1e-9 else 0.0

             # Calculate vector end points for this frame using the scale from the slider
             vel_end_x = current_x + current_vx * velocity_vector_scale
             vel_end_y = current_y + current_vy * velocity_vector_scale
             force_end_x = current_x
             force_end_y = current_y - current_force_mag * force_vector_scale # Force is down in y-up coords

             # Define the data update for this frame
             # Update the 'Trayectoria Trazada', 'Proyectil', 'Velocidad', and 'Fuerza Neta' traces
             frame_data = [
                 # Update Trajectory trace: include all points up to the current index
                 go.Scatter(x=history_x[:i+1], y=history_y[:i+1], mode='lines', name='Trayectoria Trazada',
                            line=dict(color='green', width=2)), # Ensure color/width match base trace

                 # Update Proyectil trace
                 go.Scatter(x=[current_x], y=[current_y], mode='markers', name='Proyectil',
                            marker=dict(color='red', size=10)),

                 # Update Velocidad trace
                 go.Scatter(x=[current_x, vel_end_x], y=[current_y, vel_end_y], mode='lines', name='Velocidad',
                            line=dict(color='blue', width=3)),

                 # Update Fuerza Neta trace
                 go.Scatter(x=[current_x, force_end_x], y=[current_y, force_end_y], mode='lines', name='Fuerza Neta',
                            line=dict(color='red', width=3)) # Ensure color/width match base trace
             ]

             # Add frame to the list
             frames.append(go.Frame(data=frame_data, name=str(i)))


         # Add the frames to the figure
         fig_sim.frames = frames

         # --- Add Animation Controls (Layout Configuration) ---
         steps = []
         if history_time:
             steps = [dict(
                 method='animate',
                 args=[[str(k)], dict(mode='immediate', frame=dict(duration=frame_duration_ms, redraw=True), transition=dict(duration=0))],
                 label=f'{history_time[k]:.2f} s'
             ) for k in range(len(history_time))]

         sliders = [dict(
             steps=steps,
             transition=dict(duration=0),
             x=0.1, y=0,
             currentvalue=dict(font=dict(size=12), prefix='Tiempo: ', visible=True, xanchor='right'),
             len=0.9
         )]

         updatemenus = [dict(
             type='buttons',
             buttons=[dict(
                 label='Play',
                 method='animate',
                 args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate')]
             ), dict(
                 label='Pause',
                 method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')]
             )],
             direction='left',
             pad=dict(r=10, t=87),
             showactive=False,
             x=0.1, xanchor='right', y=0, yanchor='top'
         )]

         # Calculate appropriate range for the plot including space for vectors
         max_x_needed = max(history_x) if history_x else preds['max_range_m']
         max_x_needed = max(max_x_needed, preds['max_range_m'] * 1.1 if preds['max_range_m'] > 0 else 0)
         # Add buffer based on initial vector lengths scaled by their respective factors
         max_x_needed = max(max_x_needed, initial_velocity * velocity_vector_scale + 10)
         max_x_plot = max(10, max_x_needed * 1.1)

         max_y_needed = max(history_y) if history_y else preds['max_height_m']
         max_y_needed = max(max_y_needed, preds['max_height_m'] * 1.1 if preds['max_height_m'] > 0 else 0)
         max_y_needed = max(max_y_needed, initial_height_m)
          # Add buffer based on initial upward velocity vector height
         max_y_needed = max(max_y_needed, initial_height_m + (initial_velocity * math.sin(math.radians(angle_degrees))) * velocity_vector_scale + 5)

         min_y_needed = min(history_y) if history_y else min(initial_height_m, 0)
         # Add buffer based on initial downward force vector depth
         min_y_needed = min(min_y_needed, initial_height_m - (mass_kg * G * force_vector_scale) - 5 if G > 1e-9 else min_y_needed)

         max_y_plot = max(initial_height_m + 10, max_y_needed + 5)
         min_y_plot = min(min_y_needed - 5, -10)


         fig_sim.update_layout(
             title='Simulación Animada de Proyectil',
             xaxis_title='Posición X (m)',
             yaxis_title='Posición Y (m)',
             yaxis=dict(
                 scaleanchor="x",
                 scaleratio=1,
                 range=[min_y_plot, max_y_plot]
             ),
             xaxis=dict(
                 range=[min(0, -max_x_plot*0.05), max_x_plot]
             ),
             showlegend=True,
             hovermode='closest',
             updatemenus=updatemenus,
             sliders=sliders
         )

         # Add a shape for the ground
         fig_sim.add_shape(type="line",
                           x0=min(0, -max_x_plot*0.05), y0=0, x1=max_x_plot, y1=0,
                           line=dict(color="gray", width=2),
                           xref='x', yref='y'
                           )
         # Add a shape for the Y axis
         fig_sim.add_shape(type="line",
                           x0=0, y0=min_y_plot, x1=0, y1=max_y_plot,
                           line=dict(color="gray", width=2),
                           xref='x', yref='y'
                           )

         st.plotly_chart(fig_sim, use_container_width=True)

         st.info("Nota: Los vectores de Velocidad (azul) y Fuerza Neta (rojo) se mueven con el proyectil en la animación. Las puntas de flecha no se animan fácilmente.")


    else:
         st.info("Ajusta los parámetros en la barra lateral y haz clic en 'Lanzar / Recalcular Simulación' para ver la animación.")


# --- Gráficos Históricos ---
st.subheader("Gráficos de Movimiento vs Tiempo")

if st.session_state.simulating or len(st.session_state.history_time) > 1:
    history_data = {
        "Tiempo (s)": st.session_state.history_time,
        "Posición X (m)": st.session_state.history_x_pos,
        "Posición Y (m)": st.session_state.history_y_pos,
        "Velocidad X (m/s)": st.session_state.history_vx,
        "Velocidad Y (m/s)": st.session_state.history_vy,
        "Aceleración X (m/s²)": st.session_state.history_ax,
        "Aceleración Y (m/s²)": st.session_state.history_ay,
        "Magnitud Fuerza Neta (N)": st.session_state.history_f_net_mag,
    }

    graph_cols = st.columns(2)

    graph_titles = [
        "Posición X vs Tiempo", "Posición Y vs Tiempo",
        "Velocidad X vs Tiempo", "Velocidad Y vs Tiempo",
        "Aceleración X vs Tiempo", "Aceleración Y vs Tiempo",
        "Magnitud Fuerza Neta vs Tiempo",
    ]
    graph_keys = [
        "Posición X (m)", "Posición Y (m)",
        "Velocidad X (m/s)", "Velocidad Y (m/s)",
        "Aceleración X (m/s²)", "Aceleración Y (m/s²)",
        "Magnitud Fuerza Neta (N)",
    ]

    for i, title in enumerate(graph_titles):
        col = graph_cols[i % 2]
        with col:
            key = graph_keys[i]
            fig_hist = go.Figure(data=go.Scatter(x=history_data["Tiempo (s)"], y=history_data[key], mode='lines'))
            fig_hist.update_layout(
                title=title,
                xaxis_title="Tiempo (s)",
                yaxis_title=key,
                hovermode='x unified'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

st.write("---")
st.write("Simulador básico de proyectiles sin resistencia del aire.")
st.write("Desarrollado con Python, Streamlit y Plotly.")