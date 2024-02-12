"""
Particle simulation made with numpy.

Inspired by "A remarkable periodic solution of the three-body problem
in the case of equal masses", A. Chenciner and R. Montgomery, 2000,
https://arxiv.org/pdf/math/0011268.pdf
"""

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from tqdm import tqdm

pv.set_plot_theme("dark")

R = 0.05
M = 1.0
G = 0.01 * 100

POS = [
    (0.97000436, -0.24308753, 0.0),
    (-0.97000436, 0.24308753, 0.0),
    (0.0, 0.0, 0.0),
]
VEL = [
    (0.4662036850, 0.4323657300, 0.0),
    (0.4662036850, 0.4323657300, 0.0),
    (-0.93240737, -0.86473146, 0.0),
]
N_PARTS = len(POS)

STIFFNESS = 200
DRAG_COEF = 0.3

TIME_STEP = 0.00005
END_TIME = 2 * 6.32591398
FPS = 59
N_FRAMES = int(FPS * END_TIME)
N_ITER = END_TIME / TIME_STEP
N_SUBITER = int(N_ITER / N_FRAMES)

WINDOW_SIZE = (800, 400)
TRAIL_LENGTH = 30


def iteration():
    particles["force"].fill(0.0)

    for i, part_i in enumerate(particles):
        for j, part_j in enumerate(particles[i + 1 :], start=i + 1):
            rel_pos = part_j["pos"] - part_i["pos"]
            dist = np.linalg.norm(rel_pos)
            normal = rel_pos / dist

            gravitational = G * part_i["mass"] * part_j["mass"] / dist**2 * normal
            contact = np.array((0.0, 0.0, 0.0))
            if (overlap := part_i["radius"] + part_j["radius"] - dist) > 0:
                rel_vel = part_j["vel"] - part_i["vel"]

                spring = -STIFFNESS * overlap
                drag_force = DRAG_COEF * np.dot(rel_vel, normal)

                contact = (spring + drag_force) * normal

            force = gravitational + contact

            particles[i]["force"] += force
            particles[j]["force"] -= force

    particles["momentum"] += particles["force"] * TIME_STEP
    particles["vel"] = particles["momentum"] * particles["inv_mass"][:, np.newaxis]
    particles["pos"] += particles["vel"] * TIME_STEP


def initialize_plot(
    cmap: str = "tab10",
    plot_momentum: bool = True,
    gif_filename: Optional[str] = None,
):
    colormap = plt.get_cmap(cmap)

    plotter = pv.Plotter(
        shape=(1, 1 + plot_momentum),
        window_size=WINDOW_SIZE,
        off_screen=True,
    )

    plotter.subplot(0, 0)
    for i, part in enumerate(particles):
        plotter.add_mesh(
            pv.Sphere(center=part["pos"], radius=part["radius"]),
            name=str(i),
        )
    plotter.view_xy()

    if plot_momentum:
        fig = initialize_figure()

        plotter.subplot(0, 1)
        charts = pv.ChartMPL(fig)
        plotter.add_chart(charts)

    if gif_filename:
        plotter.open_gif(gif_filename, fps=FPS, subrectangles=True)

    return plotter, colormap


def update_plot(
    frame: int,
    plotter: pv.Plotter,
    colormap: plt.cm.ScalarMappable,
    plot_momentum: bool = True,
):
    trails[frame] = particles["pos"]

    momentum[frame] = particles["momentum"][:, 0]
    angular_momentum[frame] = np.cross(
        particles["pos"],
        particles["momentum"],
    )[:, 2]
    elapsed_time[frame] = frame * N_SUBITER * TIME_STEP

    plotter.subplot(0, 0)
    for i, particle in enumerate(particles):
        color = colormap(i, N_PARTS)
        plotter.add_mesh(
            pv.Sphere(
                center=particle["pos"],
                radius=particle["radius"],
            ),
            name=str(i),
            color=color,
        )
        line = pv.lines_from_points(
            trails[max(0, frame - TRAIL_LENGTH + 1) : frame + 1, i]
        )
        plotter.add_mesh(
            line,
            color=color,
            line_width=1,
            name=f"trail_{i}",
        )

    if plot_momentum:
        plotter.subplot(0, 1)
        axes = plotter.renderer._charts._charts[0]._fig.axes
        update_figure(axes)

    if hasattr(plotter, "_gif_filename"):
        plotter.write_frame()


def initialize_figure() -> plt.Figure:
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        tight_layout=True,
        figsize=(10, 8),
    )
    fig.suptitle("Momentum conservation")

    for i in range(N_PARTS):
        color = colormap(i, N_PARTS)
        ax[0].plot(
            elapsed_time,
            momentum[:, i],
            label=f"Particle {i + 1}",
            color=color,
        )
        ax[1].plot(
            elapsed_time,
            angular_momentum[:, i],
            label=f"Particle {i + 1}",
            color=color,
        )

    ax[0].plot(
        elapsed_time,
        np.sum(momentum, axis=1),
        label="Total momentum",
        color=colormap(N_PARTS, N_PARTS),
    )
    ax[1].plot(
        elapsed_time,
        np.sum(angular_momentum, axis=1),
        label="Total angular momentum",
        color=colormap(N_PARTS, N_PARTS),
    )
    ax[0].set(
        title="Momentum",
        xlabel="time (s)",
        ylabel="Momentum X",
        xlim=(0, END_TIME),
        # ylim=(-0.03, 0.02),
    )
    ax[0].legend()

    ax[1].set(
        title="Angular Momentum",
        xlabel="time (s)",
        ylabel="Angular Momentum Z",
        xlim=(0, END_TIME),
        # ylim=(-0.0015, 0.002),
    )
    ax[1].legend()

    return fig


def update_figure(axes: list[plt.Axes]):
    mom_ax, ang_mom_ax = axes
    for i in range(N_PARTS):
        mom_ax.get_lines()[i].set_data(
            elapsed_time[:frame],
            momentum[:frame, i],
        )
        ang_mom_ax.get_lines()[i].set_data(
            elapsed_time[:frame],
            angular_momentum[:frame, i],
        )
    mom_ax.get_lines()[-1].set_data(
        elapsed_time[:frame],
        np.sum(momentum[:frame], axis=1),
    )
    ang_mom_ax.get_lines()[-1].set_data(
        elapsed_time[:frame],
        np.sum(angular_momentum[:frame], axis=1),
    )


def initialize_particles():
    particle_type = np.dtype(
        [
            ("pos", np.float64, 3),
            ("vel", np.float64, 3),
            ("acc", np.float64, 3),
            ("force", np.float64, 3),
            ("momentum", np.float64, 3),
            ("radius", np.float64),
            ("mass", np.float64),
            ("inv_mass", np.float64),
        ]
    )
    particles = np.zeros(shape=N_PARTS, dtype=particle_type)
    particles["pos"] = POS
    particles["vel"] = VEL
    particles["radius"] = R
    particles["mass"] = M
    particles["inv_mass"] = 1.0 / particles["mass"]
    particles["momentum"] = particles["mass"][:, np.newaxis] * particles["vel"]
    return particles


particles = initialize_particles()
trails = np.zeros(shape=(N_FRAMES, N_PARTS, 3), dtype=float)

elapsed_time = np.zeros(shape=(N_FRAMES,), dtype=float)
momentum = np.zeros(shape=(N_FRAMES, N_PARTS), dtype=float)
angular_momentum = np.zeros_like(momentum)

plotter, colormap = initialize_plot(plot_momentum=False, gif_filename="particles.gif")

for frame in tqdm(range(N_FRAMES), unit="frame"):
    for _ in range(N_SUBITER):
        iteration()
    update_plot(frame, plotter, colormap, plot_momentum=False)

if hasattr(plotter, "_gif_filename"):
    plotter.close()
