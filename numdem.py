"""
Particle simulation made with numpy.

Inspired from the video of Rhett Allain 'Dot Physics':
https://www.youtube.com/watch?v=jEaAaqgw2tA
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

R = 0.01
M = 0.1
V_0 = -0.15
G = 0.01

STIFFNESS = 200
DRAG_COEF = 0.3

TIME_STEP = 0.001

N_FRAMES = 100
N_SUBITER = 30
N_ITER = N_FRAMES * N_SUBITER

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
pos = [
    (0.0, 0.0, 0.0),
    (4 * R, 2 * R, 0.0),
    # (-3 * R, 0.0, 0.0),
    # (6 * R, 0.0, 0.0),
    # (0.0, 0.0, 4.0 * R),
]
particles = np.zeros(shape=len(pos), dtype=particle_type)
particles["pos"] = pos
particles["vel"][1] = (V_0, 0.0, 0.0)
particles["radius"] = R
particles["mass"] = M
particles["inv_mass"] = 1.0 / particles["mass"]
particles["momentum"] = particles["mass"][:, np.newaxis] * particles["vel"]


elapsed_time = np.zeros(shape=(N_FRAMES,), dtype=float)
momentum = np.zeros(shape=(N_FRAMES, len(particles)), dtype=float)
angular_momentum = np.zeros(shape=(N_FRAMES, len(particles)), dtype=float)

fig, ax = plt.subplots(nrows=2, ncols=1, tight_layout=True, figsize=(10, 8))
fig.suptitle("Momentum conservation")

plots: dict[str, list[plt.Line2D]] = {"momentum": [], "angular": []}
for i in range(len(particles)):
    plots["momentum"].append(
        ax[0].plot(elapsed_time, momentum[:, i], label=f"Particle {i + 1}")[0]
    )
    plots["angular"].append(
        ax[1].plot(
            elapsed_time,
            angular_momentum[:, i],
            label=f"Particle {i + 1}",
        )[0]
    )
tot_mom = ax[0].plot(
    elapsed_time,
    np.sum(momentum, axis=1),
    label="Total momentum",
)[0]
tot_ang_mom = ax[1].plot(
    elapsed_time,
    np.sum(angular_momentum, axis=1),
    label="Total angular momentum",
)[0]
ax[0].set(
    title="Momentum",
    xlabel="time (s)",
    ylabel="Momentum X",
    xlim=(0, N_ITER * TIME_STEP),
    ylim=(-0.03, 0.02),
)
ax[0].legend()

ax[1].set(
    title="Angular Momentum",
    xlabel="time (s)",
    ylabel="Angular Momentum Z",
    xlim=(0, N_ITER * TIME_STEP),
    ylim=(-0.0015, 0.002),
)
ax[1].legend()

pl = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
pl.subplot(0, 0)
pl.view_xy()

pl.subplot(0, 1)
charts = pv.ChartMPL(fig)
pl.add_chart(charts)

pl.open_gif("particles.gif")
for frame in range(N_FRAMES):
    for _ in range(N_SUBITER):
        particles["force"].fill(0.0)

        for i, part_i in enumerate(particles):
            for j, part_j in enumerate(particles[i + 1 :], start=i + 1):
                rel_pos = part_j["pos"] - part_i["pos"]
                dist = np.linalg.norm(rel_pos)
                normal = rel_pos / dist

                gravitational = (
                    G * particles[i]["mass"] * particles[j]["mass"] / dist**2 * normal
                )
                contact = np.array((0.0, 0.0, 0.0))
                if (
                    overlap := particles[i]["radius"] + particles[j]["radius"] - dist
                ) > 0:
                    rel_vel = particles[j]["vel"] - particles[i]["vel"]

                    spring = -STIFFNESS * overlap
                    drag_force = DRAG_COEF * np.dot(rel_vel, normal)

                    contact = (spring + drag_force) * normal

                force = gravitational + contact

                particles[i]["force"] += force
                particles[j]["force"] -= force

        particles["momentum"] += particles["force"] * TIME_STEP
        particles["vel"] = particles["momentum"] * particles["inv_mass"][:, np.newaxis]
        particles["pos"] += particles["vel"] * TIME_STEP

    momentum[frame] = particles["momentum"][:, 0]
    angular_momentum[frame] = np.cross(particles["pos"], particles["momentum"])[:, 2]
    elapsed_time[frame] = frame * N_SUBITER * TIME_STEP

    pl.subplot(0, 0)
    for i, particle in enumerate(particles):
        pl.add_mesh(
            pv.Sphere(center=particle["pos"], radius=particle["radius"]), name=str(i)
        )

    pl.subplot(0, 1)
    for i in range(len(particles)):
        plots["momentum"][i].set_data(elapsed_time[:frame], momentum[:frame, i])
        plots["angular"][i].set_data(elapsed_time[:frame], angular_momentum[:frame, i])
    tot_mom.set_data(elapsed_time[:frame], np.sum(momentum[:frame], axis=1))
    tot_ang_mom.set_data(elapsed_time[:frame], np.sum(angular_momentum[:frame], axis=1))

    pl.write_frame()
pl.close()
