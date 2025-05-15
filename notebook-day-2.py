import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell(hide_code=True)
def _():
    import scipy
    import scipy.integrate as sci

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, mpl, np, plt, scipy, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return (R,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell(hide_code=True)
def _(FFMpegWriter, FuncAnimation, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100
        fps = 30 # Number of frames per second

        def animate(frame_index):    
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()

            pbar.update(1)

        pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")

    _filename = "wave_animation.mp4"
    make_video(_filename)
    mo.show_code(mo.video(src=_filename))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Constants

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    g = 1.0
    M = 1.0
    l = 1
    return M, g, l


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    f_x & = -f \sin (\theta + \phi) \\
    f_y & = +f \cos(\theta +\phi)
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    M \ddot{x} & = -f \sin (\theta + \phi) \\
    M \ddot{y} & = +f \cos(\theta +\phi) - Mg
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Moment of inertia

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell
def _(M, l):
    J = M * l * l / 3
    J
    return (J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    J \ddot{\theta} = - \ell (\sin \phi)  f
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Simulation

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell(hide_code=True)
def _(J, M, g, l, np, scipy):
    def redstart_solve(t_span, y0, f_phi):
        def fun(t, state):
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)
            d2x = (-f * np.sin(theta + phi)) / M
            d2y = (+ f * np.cos(theta + phi)) / M - g
            d2theta = (- l * np.sin(phi)) * f / J
            return np.array([dx, d2x, dy, d2y, dtheta, d2theta])
        r = scipy.integrate.solve_ivp(fun, t_span, y0, dense_output=True)
        return r.sol
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(l, np, plt, redstart_solve):
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0) = - 2*\ell$,  can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    % y(t)
    y(t)
    = \frac{2(5-\ell)}{125}\,t^3
      + \frac{3\ell-10}{25}\,t^2
      - 2\,t
      + 10
    $$

    $$
    % f(t)
    f(t)
    = M\!\Bigl[
        \frac{12(5-\ell)}{125}\,t
        + \frac{6\ell-20}{25}
        + g
      \Bigr].
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(M, g, l, np, plt, redstart_solve):

    def smooth_landing_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example()
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):
    def smooth_landing_example_force():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example_force()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell(hide_code=True)
def _(M, R, g, l, mo, mpl, np, plt):
    def draw_booster(x=0, y=l, theta=0.0, f=0.0, phi=0.0, axes=None, **options):
        L = 2 * l
        if axes is None:
            _fig, axes = plt.subplots()

        axes.set_facecolor('#F0F9FF') 

        ground = np.array([[-2*l, 0], [2*l, 0], [2*l, -l], [-2*l, -l], [-2*l, 0]]).T
        axes.fill(ground[0], ground[1], color="#E3A857", **options)

        b = np.array([
            [l/10, -l], 
            [l/10, l], 
            [0, l+l/10], 
            [-l/10, l], 
            [-l/10, -l], 
            [l/10, -l]
        ]).T
        b = R(theta) @ b
        axes.fill(b[0]+x, b[1]+y, color="black", **options)

        ratio = l / (M*g) # when f= +MG, the flame length is l 

        flame = np.array([
            [l/10, 0], 
            [l/10, - ratio * f], 
            [-l/10, - ratio * f], 
            [-l/10, 0], 
            [l/10, 0]
        ]).T
        flame = R(theta+phi) @ flame
        axes.fill(
            flame[0] + x + l * np.sin(theta), 
            flame[1] + y - l * np.cos(theta), 
            color="#FF4500", 
            **options
        )

        return axes

    _axes = draw_booster(x=0.0, y=20*l, theta=np.pi/8, f=M*g, phi=np.pi/8)
    _fig = _axes.figure
    _axes.set_xlim(-4*l, 4*l)
    _axes.set_ylim(-2*l, 24*l)
    _axes.set_aspect("equal")
    _axes.grid(True)
    _MaxNLocator = mpl.ticker.MaxNLocator
    _axes.xaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.yaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.set_axisbelow(True)
    mo.center(_fig)
    return (draw_booster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualisation

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin the with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell(hide_code=True)
def _(draw_booster, l, mo, np, plt, redstart_solve):
    def sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_1()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_2():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_2()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_3()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_4():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_4()
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_1.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_1())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_2():
        L = 2*l

        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_2.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_2())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_3.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_3())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_4():
        L = 2*l
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_4.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_4())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linearized Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Equilibria

    We assume that $|\theta| < \pi/2$, $|\phi| < \pi/2$ and that $f > 0$. What are the possible equilibria of the system for constant inputs $f$ and $\phi$ and what are the corresponding values of these inputs?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        À l'équilibre, on impose :

        $$
        \ddot{x} = 0, \quad \ddot{y} = 0, \quad \ddot{\theta} = 0
        $$

        Équations dynamiques :

        $$
        M \ddot{x} = -f \sin(\theta + \phi)
        $$

        $$
        M \ddot{y} = f \cos(\theta + \phi) - Mg
        $$

        $$
        J \ddot{\theta} = -\ell \sin(\phi) f
        $$

        Mise à zéro des dérivées temporelles :

        $$
        -f \sin(\theta + \phi) = 0
        $$

        $$
        f \cos(\theta + \phi) - Mg = 0
        $$

        $$
        -\ell \sin(\phi) f = 0
        $$

        Raisonnement :

        $$
        \theta + \phi = 0 \quad \text{(car } \sin(\theta + \phi) = 0 \text{ et } |\theta|, |\phi| < \frac{\pi}{2})
        $$

        $$
        \phi = 0 \Rightarrow \theta = 0
        $$

        $$
        f = Mg
        $$

        Donc, à l’équilibre :

        $$
        \theta = 0, \quad \phi = 0, \quad f = Mg, \quad \dot{x} = \dot{y} = \dot{\theta} = 0
        $$

        $$x = 0,\quad y = l $$

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linearized Model

    Introduce the error variables $\Delta x$, $\Delta y$, $\Delta \theta$, and $\Delta f$ and $\Delta \phi$ of the state and input values with respect to the generic equilibrium configuration.
    What are the linear ordinary differential equations that govern (approximately) these variables in a neighbourhood of the equilibrium?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        On définit les erreurs :

        $$
        \Delta x = x - x_0, \quad \Delta y = y - y_0, \quad \Delta \theta = \theta, \quad \Delta \phi = \phi, \quad \Delta f = f - Mg
        $$

        #### Linéarisation de $\ddot{x}$ :

        $$
        M \ddot{x} = -f \sin(\theta + \phi)
        $$

        Petits angles $\Rightarrow \sin(\theta + \phi) \approx \theta + \phi$ :

        $$
        M \Delta \ddot{x} \approx -(Mg + \Delta f)(\Delta \theta + \Delta \phi) \approx -Mg (\Delta \theta + \Delta \phi)
        $$

        $$
        \Delta \ddot{x} = -g (\Delta \theta + \Delta \phi)
        $$

        #### Linéarisation de $\ddot{y}$ :

        $$
        M \ddot{y} = f \cos(\theta + \phi) - Mg
        $$

        Petits angles $\Rightarrow \cos(\theta + \phi) \approx 1 - \frac{(\Delta \theta + \Delta \phi)^2}{2}$ :

        $$
        M \Delta \ddot{y} \approx \Delta f - Mg \cdot \frac{(\Delta \theta + \Delta \phi)^2}{2} \approx \Delta f
        $$

        $$
        \Delta \ddot{y} = \frac{\Delta f}{M}
        $$

        #### Linéarisation de $\ddot{\theta}$ :

        $$
        J \ddot{\theta} = -\ell \sin(\phi) f \approx -\ell \Delta \phi (Mg + \Delta f) \approx -\ell Mg \Delta \phi
        $$

        $$
        \Delta \ddot{\theta} = -\frac{\ell Mg}{J} \Delta \phi
        $$

        ### Équations linéarisées finales :

        $$
        \Delta \ddot{x} = -g(\Delta \theta + \Delta \phi)
        $$

        $$
        \Delta \ddot{y} = \frac{\Delta f}{M}
        $$

        $$
        \Delta \ddot{\theta} = -\frac{\ell Mg}{J} \Delta \phi
        $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Standard Form

    What are the matrices $A$ and $B$ associated to this linear model in standard form?
    Define the corresponding NumPy arrays `A` and `B`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Mise en forme standard du système (modèle d'état)

        On définit le **vecteur d'état** :

        $$
        X = \begin{bmatrix}
            \Delta x \\
            \Delta \dot{x} \\
            \Delta y \\
            \Delta \dot{y} \\
            \Delta \theta \\
            \Delta \dot{\theta}
        \end{bmatrix}
        $$

        Et le **vecteur de commande** :

        $$
        U = \begin{bmatrix}
            \Delta f \\
            \Delta \phi
        \end{bmatrix}
        $$

        La dynamique du système s’écrit sous la forme espace d’état :

        $$
        \dot{X} = A X + B U
        $$

        #### Matrice $A$ :

        $$
        A =
        \begin{bmatrix}
        0 & 1 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & -g & 0 \\
        0 & 0 & 0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 1 \\
        0 & 0 & 0 & 0 & 0 & 0
        \end{bmatrix}
        $$

        #### Matrice $B$ :

        $$
        B =
        \begin{bmatrix}
        0 & 0 \\
        0 & -g \\
        0 & 0 \\
        \frac{1}{M} & 0 \\
        0 & 0 \\
        0 & -\frac{\ell M g}{J}
        \end{bmatrix}
        $$

        Avec :

        $$
        J = \frac{M \ell^2}{3}
        $$

        où :

        - $g$ : gravité
        - $M$ : masse du booster
        - $\ell$ : distance bras de levier
    """
    )
    return


@app.cell
def _(J, M, g, l, np):

    A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, -g, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])


    B = np.array([
        [0, 0],
        [0, -g],
        [0, 0],
        [1/M, 0],
        [0, 0],
        [0, -l*M*g/J]
    ])

    return A, B


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To determine stability, we examine the eigenvalues of A:""")
    return


@app.cell
def _(A, np):
    eigenvalues = np.linalg.eigvals(A)
    print("Eigenvalues:", eigenvalues)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les valeurs propres de la matrice $A$ sont :

    $$
    \lambda =
    \begin{bmatrix}
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    $$

        Toutes les valeurs propres sont nulles, c’est-à-dire **situées sur l’axe imaginaire** (en fait à l’origine). Cela implique :

        - Les valeurs propres **ne sont pas strictement dans le demi-plan gauche** du plan complexe.
        - Le système **n’est pas asymptotiquement stable**.
        - Le système est **au mieux marginalement stable**.
    

        À l’équilibre, si la fusée n’est **pas perturbée**, elle y reste. En revanche :

        - Une **petite perturbation** ne ramène pas la fusée à l’équilibre.
        - Elle aura tendance à **dériver**.
        - Cela est dû à la présence de **double intégrateurs** dans la dynamique.

        Donc, le système **n’a pas de dynamique de rappel** pour revenir à l’équilibre après une perturbation.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """
    )
    return


@app.cell
def _(A, B, np):
    def compute_controllability_matrix(A, B):
        n = A.shape[0]  
        m = B.shape[1]  
        C = np.zeros((n, n*m))
    
        A_power = np.eye(n)
        for i in range(n):
            C[:, i*m:(i+1)*m] = A_power @ B
            A_power = A_power @ A
        
        return C

    C = compute_controllability_matrix(A, B)
    rank_C = np.linalg.matrix_rank(C)
    print(f"Rank of controllability matrix: {rank_C}")
    print(f"System is {'controllable' if rank_C == A.shape[0] else 'not controllable'}")
    return (compute_controllability_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Le rang de la matrice de contrôlabilité est égal à 6, ce qui correspond à la dimension du système.
    Par conséquent, le système est contrôlable.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Lateral Dynamics

    We limit our interest in the lateral position $x$, the tilt $\theta$ and their derivatives (we are for the moment fine with letting $y$ and $\dot{y}$ be uncontrolled). We also set $f = M g$ and control the system only with $\phi$.

    What are the new (reduced) matrices $A$ and $B$ for this reduced system?
    Check the controllability of this new system.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Modèle réduit (position latérale et inclinaison)

    On se concentre sur la position latérale \( x \), l’inclinaison \( \theta \) et leurs dérivées.  
    La poussée est fixée à \( f = Mg \) et on contrôle uniquement avec \( \phi \).

    Le vecteur d’état réduit est :

    \[
    X_\text{réduit} = 
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}
    \]

    Le vecteur d’entrée réduit :

    \[
    U_\text{réduit} = \begin{bmatrix}
    \Delta \phi
    \end{bmatrix}
    \]

    La matrice \( A \) réduite :

    \[
    A_\text{réduit} =
    \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix}
    \]

    La matrice \( B \) réduite :

    \[
    B_\text{réduit} =
    \begin{bmatrix}
    0 \\
    -g \\
    0 \\
    -\dfrac{\ell M g}{J}
    \end{bmatrix}
    \]

    """
    )
    return


@app.cell
def _(J, M, compute_controllability_matrix, g, l, np):
    A_reduced = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    B_reduced = np.array([
        [0],
        [-g],
        [0],
        [-l*M*g/J]
    ]).reshape(4, 1)


    C_reduced = compute_controllability_matrix(A_reduced, B_reduced)
    rank_C_reduced = np.linalg.matrix_rank(C_reduced)
    print(f"Rank of reduced controllability matrix: {rank_C_reduced}")
    print(f"Reduced system is {'controllable' if rank_C_reduced == A_reduced.shape[0] else 'not controllable'}")
    return A_reduced, B_reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Le rang de la matrice de contrôlabilité réduite est 4, ce qui est égal à la dimension du système réduit. Le système réduit est donc également contrôlable.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linear Model in Free Fall

    Make graphs of $y(t)$ and $\theta(t)$ for the linearized model when $\phi(t)=0$,
    $x(0)=0$, $\dot{x}(0)=0$, $\theta(0) = 45 / 180  \times \pi$  and $\dot{\theta}(0) =0$. What do you see? How do you explain it?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Dans ce scénario, on analyse le modèle linéarisé en supposant que \(\phi(t) = 0\), avec les conditions initiales suivantes :

    - \( x(0) = 0 \)  
    - \( \dot{x}(0) = 0 \)  
    - \( \theta(0) = 45^\circ = \frac{\pi}{4} \)  
    - \( \dot{\theta}(0) = 0 \)

    Pour simuler ce système, on peut utiliser les méthodes standards de résolution d’équations différentielles (ODE), comme :
    """
    )
    return


@app.cell
def _(A_reduced, B_reduced, np, plt):
    from scipy.integrate import solve_ivp

    def linear_model_dynamics(t, state, A_reduced, B_reduced, u):
        return A_reduced @ state + B_reduced @ u

    x0 = np.array([0.0, 0.0, np.pi/4, 0.0])
    t_span = [0, 20]
    # Time span
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Control input (zero for free fall)
    u = np.array([0.0])


    # Solve ODE
    sol = solve_ivp(
        lambda t, x: linear_model_dynamics(t, x, A_reduced, B_reduced, u),
        t_span, x0, t_eval=t_eval
    )

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(sol.t, sol.y[0], label='Δx(t)')
    plt.grid(True)
    plt.legend()
    plt.title('Position vs Time')

    plt.subplot(2, 1, 2)
    plt.plot(sol.t, sol.y[2], label='Δθ(t)')
    plt.grid(True)
    plt.legend()
    plt.title('Angle vs Time')

    plt.tight_layout()
    plt.show()


    return solve_ivp, t_eval, t_span, x0


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    Pour le modèle linéarisé en chute libre avec une inclinaison initiale de 45° :

    - La position angulaire \( \theta \) reste constante, car aucun couple n'agit sur le lanceur lorsque \( \phi = 0 \).

    - La position \( x \) diminue de manière quadratique dans le temps.

    - Cela a du sens physiquement : en l'absence de commande et avec une inclinaison initiale, la fusée continue à s'accélèrer horizentalement, sans aucune rotation.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Manually Tuned Controller

    Try to find the two missing coefficients of the matrix 

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    manages  when
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  and $\Delta \dot{\theta}(0) =0$ to: 

      - make $\Delta \theta(t) \to 0$ in approximately $20$ sec (or less),
      - $|\Delta \theta(t)| < \pi/2$ and $|\Delta \phi(t)| < \pi/2$ at all times,
      - (but we don't care about a possible drift of $\Delta x(t)$).

    Explain your thought process, show your iterations!

    Is your closed-loop model asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Nous cherchons à déterminer les coefficients du vecteur \( K \) :

    \[
    K = [0,\, 0,\, k_3,\, k_4]
    \]

    tels que :

    - \( \Delta \theta(t) \rightarrow 0 \) en environ 20 secondes ou moins
    - \( |\Delta \theta(t)| < \frac{\pi}{2} \) et \( |\Delta \phi(t)| < \frac{\pi}{2} \) en tout temps

    En utilisant la relation triuvée précédemment

    \[
    \Delta \ddot{\theta} = -\frac{\ell M g}{J} \cdot \Delta \phi
    \]

    Avec la relation\( \Delta \phi = -K \cdot X \), on obtient :

    \[
    \Delta \ddot{\theta} = \frac{\ell M g}{J} \cdot k_3 \cdot \Delta \theta + \frac{\ell M g}{J} \cdot k_4 \cdot \Delta \dot{\theta}
    \]

    en faisant des itérations, on trouve : \( k_3 = -0.25 \)   et \( k_4 = -0.4\) 
    """
    )
    return


@app.cell
def _(A_reduced, B_reduced, np, plt, solve_ivp, t_eval, t_span, x0):
    def simulate_controlled_system(K, A_reduced, B_reduced, x0, t_span, t_eval):
        def dynamics(t, state):
            state_col = state.reshape(-1, 1)
        
            u = -K @ state
            u = np.array([[u]])  
            state_dot = A_reduced @ state_col + B_reduced @ u
            return state_dot.flatten()
    
        sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval)
        return sol

    k3 = -0.25
    k4 = -0.4
    K = np.array([0, 0, k3, k4])


    sol_controlled = simulate_controlled_system(
        K, A_reduced, B_reduced, x0, t_span, t_eval
    )


    plt.figure(figsize=(12, 10))


    plt.subplot(3, 1, 1)
    plt.plot(sol_controlled.t, sol_controlled.y[2], label='Δθ(t)')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=np.pi/2, color='k', linestyle=':', alpha=0.3)
    plt.axhline(y=-np.pi/2, color='k', linestyle=':', alpha=0.3)
    plt.grid(True)
    plt.legend()
    plt.title('Angle vs Time')
    plt.ylabel('Angle (rad)')

    plt.subplot(3, 1, 2)
    control_input = np.array([-K @ sol_controlled.y[:, i] for i in range(len(sol_controlled.t))])
    plt.plot(sol_controlled.t, control_input, label='Δφ(t)')
    plt.axhline(y=np.pi/2, color='k', linestyle=':', alpha=0.3)
    plt.axhline(y=-np.pi/2, color='k', linestyle=':', alpha=0.3)
    plt.grid(True)
    plt.legend()
    plt.title('Control Input vs Time')
    plt.ylabel('Control Input (rad)')



    plt.tight_layout()
    plt.show()

    return (simulate_controlled_system,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Pole Assignment

    Using pole assignement, find a matrix

    $$
    K_{pp} =
    \begin{bmatrix}
    ? & ? & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K_{pp} \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    satisfies the conditions defined for the manually tuned controller and additionally:

      - result in an asymptotically stable closed-loop dynamics,

      - make $\Delta x(t) \to 0$ in approximately $20$ sec (or less).

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell
def _(
    A_reduced,
    B_reduced,
    np,
    plt,
    simulate_controlled_system,
    t_eval,
    t_span,
    x0,
):
    from scipy import signal

    desired_poles = [-0.5+0.3j, -0.5-0.3j, -0.4+0.2j, -0.4-0.2j]

    K_pp = signal.place_poles(A_reduced, B_reduced, desired_poles).gain_matrix
    print("K_pp:", K_pp)

    # Simulate with the pole placement controller
    sol_pp = simulate_controlled_system(K_pp, A_reduced, B_reduced, x0, t_span, t_eval)

    # Check closed-loop stability
    A_cl_pp = A_reduced - B_reduced @ K_pp
    eigenvalues_cl_pp = np.linalg.eigvals(A_cl_pp)
    print("Closed-loop eigenvalues with pole placement:", eigenvalues_cl_pp)

    # Plot results
    plt.figure(figsize=(12, 10))

    # Position plot
    plt.subplot(3, 1, 1)
    plt.plot(sol_pp.t, sol_pp.y[0], label='Δx(t)')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.grid(True)
    plt.legend()
    plt.title('Position vs Time')
    plt.ylabel('Δx (m)')

    # Angle plot
    plt.subplot(3, 1, 2)
    plt.plot(sol_pp.t, sol_pp.y[2], label='Δθ(t)')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=np.pi/2, color='k', linestyle=':', alpha=0.3)
    plt.axhline(y=-np.pi/2, color='k', linestyle=':', alpha=0.3)
    plt.grid(True)
    plt.legend()
    plt.title('Angle vs Time')
    plt.ylabel('Angle (rad)')

    # Control input plot
    plt.subplot(3, 1, 3)
    control_input_pp = np.array([-K_pp @ sol_pp.y[:, i] for i in range(len(sol_pp.t))])
    plt.plot(sol_pp.t, control_input_pp, label='Δφ(t)')
    plt.axhline(y=np.pi/2, color='k', linestyle=':', alpha=0.3)
    plt.axhline(y=-np.pi/2, color='k', linestyle=':', alpha=0.3)
    plt.grid(True)
    plt.legend()
    plt.title('Control Input vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input (rad)')

    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Choix des pôles désirés
    on a choisi les valeurs des poles pour que le système soit résponsive le plus rapide possible ( < 20s)


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Optimal Control

    Using optimal, find a gain matrix $K_{oc}$ that satisfies the same set of requirements that the one defined using pole placement.

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell
def _(A_reduced, B_reduced, np, plt, solve_ivp):
    from scipy import linalg
    from scipy.linalg import solve_continuous_are
    from IPython.display import HTML, display
    import os

    Q = np.diag([3.0, 1.0, 20.0, 1.0]) 
    R_scalar = 5.0                     

    try:
        P = solve_continuous_are(A_reduced, B_reduced, Q, R_scalar)
        K_oc_lqr = np.linalg.inv(np.array([[R_scalar]])) @ B_reduced.T @ P
        print(f"Matrice de gain LQR K_oc = {K_oc_lqr}")
        k_x_lqr, k_vx_lqr, k_theta_lqr, k_omega_lqr = K_oc_lqr.flatten()
        print(f"  k_x_lqr = {k_x_lqr:.4f}")
        print(f"  k_vx_lqr = {k_vx_lqr:.4f}")
        print(f"  k_theta_lqr = {k_theta_lqr:.4f}")
        print(f"  k_omega_lqr = {k_omega_lqr:.4f}")

    except np.linalg.LinAlgError as e:
        print(f"La résolution LQR a échoué : {e}")
        K_oc_lqr = np.array([[0,0,0,0]]) 
        k_x_lqr, k_vx_lqr, k_theta_lqr, k_omega_lqr = K_oc_lqr.flatten()


    A_bf_lqr = A_reduced - B_reduced @ K_oc_lqr 

    val_propres_bf_lqr, _ = np.linalg.eig(A_bf_lqr)
    print(f"\nValeurs propres du système LQR en boucle fermée : {np.sort(val_propres_bf_lqr)}")

    dx0 = 0.0
    dvx0 = 0.0
    dtheta0 = (45/180) * np.pi
    domega0 = 0.0
    CI = np.array([dx0, dvx0, dtheta0, domega0])

    t_debut = 0.0
    t_fin = 30.0
    temps_eval = np.linspace(t_debut, t_fin, 1000)

    def systeme_boucle_fermee_lqr(t, dX_actuel, A_bf_matrice_lqr):
        return A_bf_matrice_lqr @ dX_actuel

    solution_bf_lqr = solve_ivp(systeme_boucle_fermee_lqr, [t_debut, t_fin], CI,
                                args=(A_bf_lqr,), dense_output=True, t_eval=temps_eval)

    dX_t_lqr = solution_bf_lqr.y
    dx_t_lqr = dX_t_lqr[0, :]
    dvx_t_lqr = dX_t_lqr[1,:]
    dtheta_t_lqr = dX_t_lqr[2, :]
    domega_t_lqr = dX_t_lqr[3, :]
    dphi_t_lqr = - (K_oc_lqr @ dX_t_lqr).flatten()

    plt.figure(figsize=(18, 10))
    plt.suptitle("Contrôle Optimal LQR", fontsize=16)

    plt.subplot(2, 2, 1)
    plt.plot(temps_eval, dtheta_t_lqr, label=r'$\Delta\theta(t)$ (LQR)')
    plt.axhline(0, color='black', linestyle='--', lw=0.8)
    plt.axhline(0.02 * dtheta0, color='gray', linestyle=':', label=r'2% de $\Delta\theta(0)$')
    plt.title(r'Erreur d\'Angle $\Delta\theta(t)$')
    plt.xlabel('Temps (s)'); plt.ylabel(r'$\Delta\theta(t)$ (radians)'); plt.grid(True); plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(temps_eval, dphi_t_lqr, label=r'$\Delta\phi(t)$ (LQR)')
    plt.axhline(np.pi/2, color='red', linestyle='--', label=r'$|\Delta\phi|_{max} = \pi/2$')
    plt.axhline(-np.pi/2, color='red', linestyle='--')
    plt.title(r'Commande d\'Entrée $\Delta\phi(t)$')
    plt.xlabel('Temps (s)'); plt.ylabel(r'$\Delta\phi(t)$ (radians)'); plt.grid(True); plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(temps_eval, dx_t_lqr, label=r'$\Delta x(t)$ (LQR)')
    plt.axhline(0, color='black', linestyle='--', lw=0.8)
    plt.title(r'Erreur de Position $\Delta x(t)$')
    plt.xlabel('Temps (s)'); plt.ylabel(r'$\Delta x(t)$ (m)'); plt.grid(True); plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(temps_eval, dvx_t_lqr, label=r'$\Delta \dot{x}(t)$ (LQR)')
    plt.axhline(0, color='black', linestyle='--', lw=0.8)
    plt.title(r'Erreur de Vitesse $\Delta \dot{x}(t)$')
    plt.xlabel('Temps (s)'); plt.ylabel(r'$\Delta \dot{x}(t)$ (m/s)'); plt.grid(True); plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    indices_stab_theta_lqr = np.where(np.abs(dtheta_t_lqr) < 0.02 * np.abs(dtheta0))[0]
    temps_stab_theta_lqr = temps_eval[indices_stab_theta_lqr[0]] if len(indices_stab_theta_lqr) > 0 else t_fin
    print(f"\nLQR : Temps de stabilisation pour Δθ (à 2%) approx : {temps_stab_theta_lqr:.2f} s")

    pic_abs_dx_lqr = np.max(np.abs(dx_t_lqr))
    idx_pic_dx_lqr = np.argmax(np.abs(dx_t_lqr))
    seuil_dx = 0.01
    indices_stab_dx_lqr = np.where(np.abs(dx_t_lqr[idx_pic_dx_lqr:]) < seuil_dx)[0]
    temps_stab_dx_lqr = temps_eval[idx_pic_dx_lqr + indices_stab_dx_lqr[0]] if len(indices_stab_dx_lqr) > 0 else t_fin
    print(f"LQR : Temps approx. pour Δx de revenir proche de 0 (seuil {seuil_dx}m) : {temps_stab_dx_lqr:.2f} s (Pic |Δx| : {pic_abs_dx_lqr:.3f}m)")

    print(f"LQR : Max |Δϕ(t)| : {np.max(np.abs(dphi_t_lqr)):.3f} rad (Limite : {np.pi/2:.3f} rad)")
    print(f"LQR : Max |Δθ(t)| : {np.max(np.abs(dtheta_t_lqr)):.3f} rad (Limite : {np.pi/2:.3f} rad)") # Devrait être dtheta0

    return (solve_continuous_are,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Pour la question sur la Commande Optimale LQR (trouver `K_oc`) :**

    1.  Nous avons aussi utilisé le modèle linéarisé réduit (`A_reduced`, `B_reduced`).
    2.  Nous avons défini une fonction de coût pénalisant les erreurs sur `Δx` et `Δθ` (via la matrice `Q`) et l'effort de commande `Δϕ` (via `R`).
    3.  L'algorithme LQR a trouvé la matrice de gains `K_oc` qui minimise cette fonction de coût, assurant une stabilisation efficace (<20s) et stable tout en équilibrant performance et usage de la commande
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


@app.cell
def _(J, M, g, l, np, plt, solve_continuous_are, solve_ivp):
    from scipy.signal import place_poles
    I = J
    c = (l * M * g) / I

    def redstart_solve_nonlinear(t_span, y0_nl, f_phi_controller_nl_func):
        def dynamics_nl(t, y_nl_current):
            x_nl, vx_nl, y_nl, vy_nl, theta_nl, omega_nl = y_nl_current
       
            # Obtenir f et phi du contrôleur
            f_thrust_nl, phi_gimbal_nl = f_phi_controller_nl_func(t, y_nl_current)

            sin_theta_nl = np.sin(theta_nl)
            cos_theta_nl = np.cos(theta_nl)
            e_theta_nl = np.array([sin_theta_nl, -cos_theta_nl])
            e_theta_perp_nl = np.array([cos_theta_nl, sin_theta_nl])

            f_exhaust_vec_nl = f_thrust_nl * (np.cos(phi_gimbal_nl) * e_theta_nl + np.sin(phi_gimbal_nl) * e_theta_perp_nl)
            F_thrust_on_rocket_nl = -f_exhaust_vec_nl
            F_net_nl = F_thrust_on_rocket_nl + np.array([0.0, -M * g])

            ddx_nl = F_net_nl[0] / M
            ddy_nl = F_net_nl[1] / M
       
            torque_nl = -l * f_thrust_nl * np.sin(phi_gimbal_nl)
            ddtheta_nl = torque_nl / I

            return [vx_nl, ddx_nl, vy_nl, ddy_nl, omega_nl, ddtheta_nl]

        sol_ivp_nl = solve_ivp(dynamics_nl, t_span, y0_nl, dense_output=True, rtol=1e-7, atol=1e-9)

        def sol_nl(t):
            t_eval_nl = np.atleast_1d(t)
            return sol_ivp_nl.sol(t_eval_nl).T
        return sol_nl


    # Pole Placement (avec omega_c_poles = 0.7)
    A_matrix_pp = np.array([ [0,1,0,0], [0,0,-g,0], [0,0,0,1], [0,0,0,0] ])
    B_col_pp = np.array([ [0],[-g],[0],[-c] ])
    omega_c_pp = 0.7
    p1_pp = omega_c_pp * (-0.3827 + 1j * 0.9239)
    p2_pp = omega_c_pp * (-0.9239 + 1j * 0.3827)
    poles_desires_pp = np.array([p1_pp, np.conj(p1_pp), p2_pp, np.conj(p2_pp)])
    K_pole_placement = place_poles(A_matrix_pp, B_col_pp, poles_desires_pp).gain_matrix
    print(f"K_pole_placement = {K_pole_placement.flatten()}")

    # LQR
    Q_lqr = np.diag([3.0, 1.0, 20.0, 1.0])
    R_lqr_scalar = 5.0
    P_lqr = solve_continuous_are(A_matrix_pp, B_col_pp, Q_lqr, R_lqr_scalar)
    K_lqr = np.linalg.inv(np.array([[R_lqr_scalar]])) @ B_col_pp.T @ P_lqr
    print(f"K_lqr = {K_lqr.flatten()}")

    x0_nl = 0.0
    vx0_nl = 0.0
    y0_nl = 10.0 # Hauteur initiale
    vy0_nl = 0.0
    theta0_nl = (45/180) * np.pi # Inclinaison initiale de 45 degrés
    omega0_nl = 0.0
    CI_non_lineaire = np.array([x0_nl, vx0_nl, y0_nl, vy0_nl, theta0_nl, omega0_nl])

    def f_phi_pp_controller(t, etat_actuel_nl, K_gain_pp):
        x_nl, vx_nl, _, _, theta_nl, omega_nl = etat_actuel_nl
   
        delta_x = x_nl
        delta_vx = vx_nl
        delta_theta = theta_nl
        delta_omega = omega_nl
   
        dX_red_actuel = np.array([delta_x, delta_vx, delta_theta, delta_omega])
   
        delta_phi_commande = - (K_gain_pp @ dX_red_actuel).item() # .item() si K_gain_pp est 1xN

        f_poussee_nl = M * g
        phi_gimbal_nl = delta_phi_commande
   
        return np.array([f_poussee_nl, phi_gimbal_nl])

    # --- Fonction de contrôleur pour LQR (utilisant K_lqr) ---
    def f_phi_lqr_controller(t, etat_actuel_nl, K_gain_lqr):
        x_nl, vx_nl, _, _, theta_nl, omega_nl = etat_actuel_nl
        delta_x = x_nl
        delta_vx = vx_nl
        delta_theta = theta_nl
        delta_omega = omega_nl
        dX_red_actuel = np.array([delta_x, delta_vx, delta_theta, delta_omega])
        delta_phi_commande = - (K_gain_lqr @ dX_red_actuel).item()
        f_poussee_nl = M * g
        phi_gimbal_nl = delta_phi_commande
        return np.array([f_poussee_nl, phi_gimbal_nl])


    # --- Simulation et Tracé ---
    t_debut_nl = 0.0
    t_fin_nl = 30.0
    temps_eval_nl = np.linspace(t_debut_nl, t_fin_nl, 1000)

    # 1. Test Pole Placement sur modèle non linéaire
    sol_nl_pp_func = redstart_solve_nonlinear(
        [t_debut_nl, t_fin_nl],
        CI_non_lineaire,
        lambda t, y_nl: f_phi_pp_controller(t, y_nl, K_pole_placement)
    )
    resultats_nl_pp = sol_nl_pp_func(temps_eval_nl)
    phi_commande_nl_pp = np.array([f_phi_pp_controller(t, resultats_nl_pp[i,:], K_pole_placement)[1] for i, t in enumerate(temps_eval_nl)])

    # 2. Test LQR sur modèle non linéaire
    sol_nl_lqr_func = redstart_solve_nonlinear(
        [t_debut_nl, t_fin_nl],
        CI_non_lineaire,
        lambda t, y_nl: f_phi_lqr_controller(t, y_nl, K_lqr)
    )
    resultats_nl_lqr = sol_nl_lqr_func(temps_eval_nl)
    phi_commande_nl_lqr = np.array([f_phi_lqr_controller(t, resultats_nl_lqr[i,:], K_lqr)[1] for i, t in enumerate(temps_eval_nl)])


    # Tracés comparatifs
    fig, axs = plt.subplots(3, 2, figsize=(18, 15), sharex=True)
    plt.suptitle("Test des Contrôleurs sur Modèle Non Linéaire", fontsize=18)

    # θ(t)
    axs[0,0].plot(temps_eval_nl, resultats_nl_pp[:,4], label='Pole Placement')
    axs[0,0].plot(temps_eval_nl, resultats_nl_lqr[:,4], label='LQR', linestyle='--')
    axs[0,0].axhline(0, color='k', lw=0.8, ls=':')
    axs[0,0].axhline(0.02 * theta0_nl, color='gray', linestyle=':', label='2% de $\theta(0)$')
    axs[0,0].set_title(r'Angle $\theta(t)$ (Non Linéaire)')
    axs[0,0].set_ylabel(r'$\theta(t)$ (rad)'); axs[0,0].grid(True); axs[0,0].legend()

    # ϕ(t)
    axs[0,1].plot(temps_eval_nl, phi_commande_nl_pp, label='Pole Placement')
    axs[0,1].plot(temps_eval_nl, phi_commande_nl_lqr, label='LQR', linestyle='--')
    axs[0,1].axhline(np.pi/2, color='r', lw=0.8, ls='--', label=r'$|\phi|_{max}=\pi/2$')
    axs[0,1].axhline(-np.pi/2, color='r', lw=0.8, ls='--')
    axs[0,1].set_title(r'Commande Cardan $\phi(t)$ (Non Linéaire)')
    axs[0,1].set_ylabel(r'$\phi(t)$ (rad)'); axs[0,1].grid(True); axs[0,1].legend()

    # x(t)
    axs[1,0].plot(temps_eval_nl, resultats_nl_pp[:,0], label='Pole Placement')
    axs[1,0].plot(temps_eval_nl, resultats_nl_lqr[:,0], label='LQR', linestyle='--')
    axs[1,0].axhline(0, color='k', lw=0.8, ls=':')
    axs[1,0].set_title(r'Position Horizontale $x(t)$ (Non Linéaire)')
    axs[1,0].set_ylabel(r'$x(t)$ (m)'); axs[1,0].grid(True); axs[1,0].legend()

    # y(t) - Juste pour voir ce qu'il advient de y
    axs[1,1].plot(temps_eval_nl, resultats_nl_pp[:,2], label='Pole Placement')
    axs[1,1].plot(temps_eval_nl, resultats_nl_lqr[:,2], label='LQR', linestyle='--')
    axs[1,1].axhline(l, color='g', lw=0.8, ls='--', label='Hauteur $y=l$ (cible du controlled landing)')
    axs[1,1].set_title(r'Position Verticale $y(t)$ (Non Linéaire)')
    axs[1,1].set_ylabel(r'$y(t)$ (m)'); axs[1,1].grid(True); axs[1,1].legend()

    # vx(t) et omega_theta(t)
    axs[2,0].plot(temps_eval_nl, resultats_nl_pp[:,1], label=r'Pole Placement $\dot{x}$')
    axs[2,0].plot(temps_eval_nl, resultats_nl_lqr[:,1], label=r'LQR $\dot{x}$', linestyle='--')
    axs[2,0].axhline(0, color='k', lw=0.8, ls=':')
    axs[2,0].set_title(r'Vitesse Horizontale $\dot{x}(t)$ (Non Linéaire)')
    axs[2,0].set_ylabel(r'$\dot{x}(t)$ (m/s)'); axs[2,0].grid(True); axs[2,0].legend()

    axs[2,1].plot(temps_eval_nl, resultats_nl_pp[:,5], label=r'Pole Placement $\dot{\theta}$')
    axs[2,1].plot(temps_eval_nl, resultats_nl_lqr[:,5], label=r'LQR $\dot{\theta}$', linestyle='--')
    axs[2,1].axhline(0, color='k', lw=0.8, ls=':')
    axs[2,1].set_title(r'Vitesse Angulaire $\dot{\theta}(t)$ (Non Linéaire)')
    axs[2,1].set_ylabel(r'$\dot{\theta}(t)$ (rad/s)'); axs[2,1].grid(True); axs[2,1].legend()


    plt.xlabel("Temps (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1.  Nous avons pris les matrices de gains `K` (obtenues via Placement de Pôles et LQR à partir du modèle linéarisé) et les avons appliquées au **modèle non linéaire complet** de la fusée.
    2.  Nous avons simulé la réponse du modèle non linéaire avec chaque contrôleur, en partant d'une inclinaison initiale de 45 degrés.
    3.  Nous avons vérifié sur les graphiques que `θ(t)` (angle) et `x(t)` (position horizontale) revenaient bien à zéro en moins de 20 secondes, et que les contraintes sur `θ` et `ϕ` (commande) étaient respectées.
    """
    )
    return


if __name__ == "__main__":
    app.run()
