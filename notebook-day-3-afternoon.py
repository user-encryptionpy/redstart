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
    return FFMpegWriter, FuncAnimation, la, mpl, np, plt, sci, scipy, tqdm


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


@app.cell
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
    $$
    \begin{bmatrix}
    x \\
    \dot{x} \\
    y \\
    \dot{y} \\
    \theta \\
    \dot{\theta} \\
    f \\
    \phi
    \end{bmatrix}
    =
    \begin{bmatrix}
    ? \\
    0 \\
    ? \\
    0 \\
    0 \\
    0 \\
    M g \\
    0
    \end{bmatrix}
    $$
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
    \begin{align*}
    M (d/dt)^2 \Delta x &= - Mg (\Delta \theta + \Delta \phi)  \\
    M (d/dt)^2 \Delta y &= \Delta f \\
    J (d/dt)^2 \Delta \theta &= - (Mg \ell) \Delta \phi \\
    \end{align*}
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
    $$
    A = 
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0  & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0  & 0 \\
    0 & 0 & 0 & 0 & 0  & 0 \\
    0 & 0 & 0 & 0 & 0  & 1 \\
    0 & 0 & 0 & 0 & 0  & 0 
    \end{bmatrix}
    \;\;\;
    B = 
    \begin{bmatrix}
    0 & 0\\ 
    0 & -g\\ 
    0 & 0\\ 
    1/M & 0\\
    0 & 0 \\
    0 & -M g \ell/J\\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(g, np):
    A = np.zeros((6, 6))
    A[0, 1] = 1.0
    A[1, 4] = -g
    A[2, 3] = 1.0
    A[4, -1] = 1.0
    A
    return (A,)


@app.cell(hide_code=True)
def _(J, M, g, l, np):
    B = np.zeros((6, 2))
    B[ 1, 1]  = -g 
    B[ 3, 0]  = 1/M
    B[-1, 1] = -M*g*l/J
    B
    return (B,)


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
def _(A, la):
    # No since 0 is the only eigenvalue of A
    eigenvalues, eigenvectors = la.eig(A)
    eigenvalues
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


@app.cell(hide_code=True)
def _(A, B, np):
    # Controllability
    cs = np.column_stack
    mp = np.linalg.matrix_power
    KC = cs([mp(A, k) @ B for k in range(6)])
    KC
    return (KC,)


@app.cell(hide_code=True)
def _(KC, np):
    # Yes!
    np.linalg.matrix_rank(KC) == 6
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


@app.cell
def _(J, M, g, l, np):
    A_lat = np.array([
        [0, 1, 0, 0], 
        [0, 0, -g, 0], 
        [0, 0, 0, 1], 
        [0, 0, 0, 0]], dtype=np.float64)
    B_lat = np.array([[0, -g, 0, - M * g * l / J]]).T

    A_lat, B_lat
    return A_lat, B_lat


@app.cell(hide_code=True)
def _(A_lat, B_lat, np):
    # Controllability
    _cs = np.column_stack
    _mp = np.linalg.matrix_power
    KC_lat = _cs([_mp(A_lat, k) @ B_lat for k in range(6)])
    KC_lat
    return (KC_lat,)


@app.cell(hide_code=True)
def _(KC_lat, np):
    np.linalg.matrix_rank(KC_lat) == 4
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
def _(J, M, g, l, np):
    def make_fun_lat(phi):
        def fun_lat(t, state):
            x, dx, theta, dtheta = state
            phi_ = phi(t, state)
            #if linearized:
            d2x = -g * (theta + phi_)
            d2theta = -M * g * l / J * phi_
            #else:
            #d2x = -g * np.sin(theta + phi_)
            #d2theta = -M * g * l / J * np.sin(phi_)
            return np.array([dx, d2x, dtheta, d2theta])

        return fun_lat
    return (make_fun_lat,)


@app.cell(hide_code=True)
def _(make_fun_lat, mo, np, plt, sci):
    def lin_sim_1():
        def _phi(t, state):
            return 0.0
        _f_lat = make_fun_lat(_phi)
        _t_span = [0, 10]
        state_0 = [0, 0, 45 * np.pi/180.0, 0]
        _r = sci.solve_ivp(fun=_f_lat, y0=state_0, t_span=_t_span, dense_output=True)
        _t = np.linspace(_t_span[0], _t_span[1], 1000)
        _sol_t = _r.sol(_t)
        _fig, (_ax1, _ax2) = plt.subplots(2, 1, sharex=True)
        _ax1.plot(_t, _sol_t[0], label=r"$x(t)$")
        _ax1.grid(True)
        _ax1.legend()
        _ax2.plot(_t, _sol_t[2], label=r"$\theta(t)$")
        _ax2.grid(True)
        _ax2.set_xlabel(r"time $t$")
        _ax2.legend()
        return mo.center(_fig)
    lin_sim_1()
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
def _(A_lat, B_lat, make_fun_lat, mo, np, plt, sci):

    def lin_sim_2():
        # Manual tuning of K (Angle only)

        K = np.array([0.0, 0.0, -1.0, -1.0])

        print("eigenvalues:", np.linalg.eig(A_lat - B_lat.reshape((-1,1)) @ K.reshape((1, -1))).eigenvalues)

        _t_span = [0, 20.0]
        _t = np.linspace(_t_span[0], _t_span[1], 1000)
        _state_0 = [0, 0, 45 * np.pi/180.0, 0]
        def _phi(t, state):
            return - K.dot(state)

        #_f_lat = make_fun_lat(_phi, linearized=False)
        #_r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        #_sol_t = _r.sol(_t)

        _f_lat = make_fun_lat(_phi) # , linearized=True)
        _r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        _sol_lin_t = _r.sol(_t)

        _fig, (_ax1, _ax2, _ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
        _ax1.plot(_t, _sol_lin_t[0], label=r"$x(t)$ (lin.)")
        #_ax1.plot(_t, _sol_t[0], label=r"$x(t)$")
        _ax1.grid(True)
        _ax1.legend(loc="lower right")
        _ax2.plot(_t, _sol_lin_t[2], label=r"$\theta(t)$ (lin.)")
        #_ax2.plot(_t, _sol_t[2], label=r"$\theta(t)$")
        _ax2.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax2.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax2.grid(True)
        _ax2.legend(loc="lower right")
        _ax3.plot(_t, _phi(_t, _sol_lin_t), label=r"$\phi(t)$ (lin.)")
        #_ax3.plot(_t, _phi(_t, _sol_t), label=r"$\phi(t)$")
        _ax3.grid(True)
        _ax3.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax3.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax3.set_xlabel(r"time $t$")
        _ax3.legend(loc="lower right")
        return mo.center(_fig)

    lin_sim_2()
    return


@app.cell
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
def _(A_lat, B_lat, make_fun_lat, mo, np, plt, sci, scipy):
    Kpp = scipy.signal.place_poles(
        A=A_lat, 
        B=B_lat, 
        poles=1.0*np.array([-0.5, -0.51, -0.52, -0.53])
    ).gain_matrix.squeeze()

    def lin_sim_3():
        print(f"Kpp = {Kpp}")

        _t_span = [0, 20.0]
        _t = np.linspace(_t_span[0], _t_span[1], 1000)
        _state_0 = [0, 0, 45 * np.pi/180.0, 0]
        def _phi(t, state):
            return - Kpp.dot(state)

        #_f_lat = make_f_lat(_phi, linearized=False)
        #_r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        #_sol_t = _r.sol(_t)

        _f_lat = make_fun_lat(_phi) # , linearized=True)
        _r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        _sol_lin_t = _r.sol(_t)

        _fig, (_ax1, _ax2, _ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
        _ax1.plot(_t, _sol_lin_t[0], label=r"$x(t)$ (lin.)")
        #_ax1.plot(_t, _sol_t[0], label=r"$x(t)$")
        _ax1.grid(True)
        _ax1.legend(loc="lower right")
        _ax2.plot(_t, _sol_lin_t[2], label=r"$\theta(t)$ (lin.)")
        #_ax2.plot(_t, _sol_t[2], label=r"$\theta(t)$")
        _ax2.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax2.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax2.grid(True)
        _ax2.legend(loc="lower right")
        _ax3.plot(_t, _phi(_t, _sol_lin_t), label=r"$\phi(t)$ (lin.)")
        #_ax3.plot(_t, _phi(_t, _sol_t), label=r"$\phi(t)$")
        _ax3.grid(True)
        _ax3.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax3.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax3.set_xlabel(r"time $t$")
        _ax3.legend(loc="lower right")
        return mo.center(_fig)

    lin_sim_3()
    return (Kpp,)


@app.cell
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
def _(A_lat, B_lat, l, make_fun_lat, mo, np, plt, sci, scipy):
    _Q = np.zeros((4,4))
    _Q[0, 0] = 1.0
    _Q[1, 1] = 0.0
    _Q[2, 2] = (2*l)**2
    _Q[3, 3] = 0.0
    _R = 10*(2*l)**2 * np.eye(1)

    _Pi = scipy.linalg.solve_continuous_are(
        a=A_lat, 
        b=B_lat, 
        q=_Q, 
        r=_R
    )
    Koc = (np.linalg.inv(_R) @ B_lat.T @ _Pi).squeeze()
    print(f"Koc = {Koc}")

    def lin_sim_4():    
        _t_span = [0, 20.0]
        _t = np.linspace(_t_span[0], _t_span[1], 1000)
        _state_0 = [0, 0, 45 * np.pi/180.0, 0]
        def _phi(t, state):
            return - Koc.dot(state)

        #_f_lat = make_fun_lat(_phi, linearized=False)
        #_r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        #_sol_t = _r.sol(_t)

        _f_lat = make_fun_lat(_phi) #, linearized=True)
        _r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        _sol_lin_t = _r.sol(_t)

        _fig, (_ax1, _ax2, _ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
        _ax1.plot(_t, _sol_lin_t[0], label=r"$x(t)$ (lin.)")
        #_ax1.plot(_t, _sol_t[0], label=r"$x(t)$")
        _ax1.grid(True)
        _ax1.legend(loc="lower right")
        _ax2.plot(_t, _sol_lin_t[2], label=r"$\theta(t)$ (lin.)")
        #_ax2.plot(_t, _sol_t[2], label=r"$\theta(t)$")
        _ax2.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax2.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax2.grid(True)
        _ax2.legend(loc="lower right")
        _ax3.plot(_t, _phi(_t, _sol_lin_t), label=r"$\phi(t)$ (lin.)")
        #_ax3.plot(_t, _phi(_t, _sol_t), label=r"$\phi(t)$")
        _ax3.grid(True)
        _ax3.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax3.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax3.set_xlabel(r"time $t$")
        _ax3.legend(loc="lower right")
        return mo.center(_fig)

    lin_sim_4()
    return (Koc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    Kpp,
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
    def video_sim_Kpp():
        t_span = [0.0, 20.0]
        y0 = [0.0, 0.0, 20.0, 0.0, 45 * np.pi/180.0, 0.0]
        def f_phi(t, state):
            x, dx, y, dy, theta, dtheta = state  
            return np.array(
                [M*g, -Kpp.dot([x, dx, theta, dtheta])]
            )
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_Kpp.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +24*l)
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

    mo.video(src=video_sim_Kpp())

    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    Koc,
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
    def video_sim_Koc():
        t_span = [0.0, 20.0]
        y0 = [0.0, 0.0, 20.0, 0.0, 45 * np.pi/180.0, 0.0]
        def f_phi(t, state):
            x, dx, y, dy, theta, dtheta = state  
            return np.array(
                [M*g, -Koc.dot([x, dx, theta, dtheta])]
            )
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_Koc.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +24*l)
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

    mo.video(src=video_sim_Koc())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Exact Linearization


    Consider an auxiliary system which is meant to compute the force $(f_x, f_y)$ applied to the booster. 

    Its inputs are 

    $$
    v = (v_1, v_2) \in \mathbb{R}^2,
    $$

    its dynamics 

    $$
    \ddot{z} = v_1 \qquad \text{ where } \quad z\in \mathbb{R}
    $$ 

    and its output $(f_x, f_y) \in \mathbb{R}^2$ is given by

    \[
    \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix} = R\left(\theta - \frac{\pi}{2}\right)
    \begin{bmatrix}
    z - M\frac{\ell}{3}\dot{\theta}^2 \\
    \frac{M\ell v_2}{3z}
    \end{bmatrix}
    \]

    ⚠️ Note that the second component $f_y$ of the reactor force is undefined whenever $z=0$.

    Consider the output $h$ of the original system

    $$
    h := 
    \begin{bmatrix}
    x - (\ell/3) \sin \theta \\
    y + (\ell/3) \cos \theta
    \end{bmatrix} \in \mathbb{R}^2
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Geometrical Interpretation

    Provide a geometrical interpretation of $h$ (for example, make a drawing).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""🔓 The coordinates $h$ represent a fixed point on the booster. Start from the reactor, move to the center of mass (distance $\ell$) then continue for $\ell/3$ in this direction. The coordinates of this point are $h$.""")
    return


@app.cell
def _(mo):
    mo.image(src="public/images/geo.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 First and Second-Order Derivatives

    Compute $\dot{h}$ as a function of $\dot{x}$, $\dot{y}$, $\theta$ and $\dot{\theta}$ (and constants) and then $\ddot{h}$ as a function of $\theta$ and $z$ (and constants) when the auxiliary system is plugged in the booster.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    🔓 We have 

    $$
    \boxed{
    \dot{h} = 
    \begin{bmatrix}
    \dot{x} - (\ell /3)  (\cos \theta) \dot{\theta} \\
    \dot{y} - (\ell /3) (\sin \theta) \dot{\theta}
    \end{bmatrix}}
    $$

    and therefore

    \begin{align*}
    \ddot{h} &=
    \begin{bmatrix}
    \ddot{x} - (\ell/3)\cos\theta\, \ddot{\theta} + (\ell/3)\sin\theta\, \dot{\theta}^2 \\
    \ddot{y} - (\ell/3)\sin\theta\, \ddot{\theta} - (\ell/3)\cos\theta\, \dot{\theta}^2
    \end{bmatrix} \\
    &=
    \begin{bmatrix}
    \frac{f_x}{M} - \frac{\ell}{3} \cos\theta \cdot \frac{3}{M\ell} (\cos\theta\, f_x + \sin\theta\, f_y) + \frac{\ell}{3} \sin\theta\, \dot{\theta}^2 \\
    \frac{f_y}{M} - g - \frac{\ell}{3} \sin\theta \cdot \frac{3}{M\ell} (\cos\theta\, f_x + \sin\theta\, f_y) - \frac{\ell}{3} \cos\theta\, \dot{\theta}^2
    \end{bmatrix} \\
    &=
    \frac{1}{M}
    \begin{bmatrix}
    \sin\theta \\
    -\cos\theta
    \end{bmatrix}
    \left(
    \begin{bmatrix}
    \sin\theta & -\cos\theta
    \end{bmatrix}
    \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix}
    + M\frac{\ell}{3} \dot{\theta}^2
    \right)
    -
    \begin{bmatrix}
    0 \\
    g
    \end{bmatrix}
    \end{align*}


    On the other hand, since

    \[
    \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix} = R\left(\theta - \frac{\pi}{2}\right)
    \begin{bmatrix}
    z - M\frac{\ell}{3}\dot{\theta}^2 \\
    \frac{M\ell v_2}{3z}
    \end{bmatrix}
    \]

    we have

    $$
    \begin{bmatrix}
    z - M\frac{\ell}{3}\dot{\theta}^2 \\
    \frac{M\ell v_2}{3z}
    \end{bmatrix}
    = R\left(\frac{\pi}{2} - \theta\right) \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix} =
    \begin{bmatrix}
    \sin \theta & - \cos \theta \\
    \cos \theta & \sin \theta
    \end{bmatrix}
    \begin{bmatrix}
    f_x \\ f_y
    \end{bmatrix}
    $$

    and therefore we end up with

    $$
    \boxed{\ddot{h} = 
      \frac{1}{M}
      \begin{bmatrix}
        \sin\theta \\
        -\cos\theta
       \end{bmatrix}
      z
      -
      \begin{bmatrix}
        0 \\
        g
      \end{bmatrix}}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Third and Fourth-Order Derivatives 

    Compute the third derivative $h^{(3)}$ of $h$ as a function of $\theta$ and $z$ (and constants) and then the fourth derivative $h^{(4)}$ of $h$ with respect to time as a function of $\theta$, $\dot{\theta}$, $z$, $\dot{z}$, $v$ (and constants) when the auxiliary system is on.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    🔓 We have 

    \[
    \boxed{
    h^{(3)} = \frac{1}{M}
    \begin{bmatrix}
    \cos \theta \\
    \sin \theta
    \end{bmatrix}
    \dot{\theta}z + \frac{1}{M}
    \begin{bmatrix}
    \sin \theta \\
    -\cos \theta
    \end{bmatrix}
    \dot{z}
    }
    \]

    and consequently

    \[
    \begin{aligned}
    h^{(4)} &= \frac{1}{M}
    \begin{bmatrix}
    -\sin \theta \\
    \cos \theta
    \end{bmatrix}
    \dot{\theta}^2 z + \frac{1}{M}
    \begin{bmatrix}
    \cos \theta \\
    \sin \theta
    \end{bmatrix}
    \frac{3}{Ml} (\cos \theta f_x + \sin \theta f_y) z \\
    &+ \frac{2}{M}
    \begin{bmatrix}
    \cos \theta \\
    \sin \theta
    \end{bmatrix}
    \dot{\theta}\dot{z} + \frac{1}{M}
    \begin{bmatrix}
    \sin \theta \\
    -\cos \theta
    \end{bmatrix}
    v_1
    \end{aligned}
    \]

    Since

    \[
    \begin{bmatrix}
    z - \frac{Ml}{3} \dot{\theta}^2 \\
    \frac{Mlv_2}{3z}
    \end{bmatrix}
    = R\left(\frac{\pi}{2} - \theta\right) \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix} =
    \begin{bmatrix}
    \sin \theta f_x - \cos \theta f_y \\
    \cos \theta f_x + \sin \theta f_y
    \end{bmatrix}
    \]

    we have

    \[
    h^{(4)} = \frac{1}{M}
    \begin{bmatrix}
    \sin \theta & \cos \theta \\
    -\cos \theta & \sin \theta
    \end{bmatrix}
    \begin{bmatrix}
    v_1 \\
    v_2
    \end{bmatrix}
    + \frac{1}{M}
    \begin{bmatrix}
    -\sin \theta \\
    \cos \theta
    \end{bmatrix}
    \dot{\theta}^2 z
    + \frac{2}{M}
    \begin{bmatrix}
    \cos \theta \\
    \sin \theta
    \end{bmatrix}
    \dot{\theta}\dot{z}
    \]

    \[
    \boxed{
    h^{(4)}
    = \frac{1}{M} R \left( \theta - \frac{\pi}{2} \right)
    \left(
    v +
    \begin{bmatrix}
    -\dot{\theta}^2 z \\
    2 \dot{\theta} \dot{z}
    \end{bmatrix}
    \right)
    }
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Exact Linearization

    Show that with yet another auxiliary system with input $u=(u_1, u_2)$ and output $v$ fed into the previous one, we can achieve the dynamics

    $$
    h^{(4)} = u
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    🔓 Since

    \[
    h^{(4)}
    = \frac{1}{M} R \left( \theta - \frac{\pi}{2} \right)
    \left(
    v +
    \begin{bmatrix}
    -\dot{\theta}^2 z \\
    2 \dot{\theta} \dot{z}
    \end{bmatrix}
    \right)
    \]  

    we can define $v$ as 

    $$
    \boxed{
    v =
    M \, R \left(\frac{\pi}{2} - \theta \right)
    u + 
    \begin{bmatrix}
    \dot{\theta}^2 z \\
    -2 \dot{\theta} \dot{z}
    \end{bmatrix}
    }
    $$

    and achieve the desired result.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 State to Derivatives of the Output

    Implement a function `T` of `x, dx, y, dy, theta, dtheta, z, dz` that returns `h_x, h_y, dh_x, dh_y, d2h_x, d2h_y, d3h_x, d3h_y`.
    """
    )
    return


@app.cell
def _(M, g, l):
    import math
    from scipy.interpolate import CubicSpline

    def T(x, dx, y, dy, theta, dtheta, z, dz):
        hx = x - (l/3)*math.sin(theta)
        hy = y - (l/3)*math.cos(theta)
        dh_x = dx - (l/3) * dtheta * math.cos(theta)
        dh_y = dy - (l/3) * dtheta * math.sin(theta)
        d2h_x = z * math.sin(theta) / M
        d2h_y = (-1) * math.cos(theta) / M - g
        d3h_x = (1/M) * (dtheta * z * math.cos(theta) + dz * math.sin(theta))
        d3h_y = (1/M) * (dtheta * z * math.sin(theta) - dz * math.cos(theta))
        return hx, hy, dh_x, dh_y, d2h_x, d2h_y, d3h_x, d3h_y

    
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Inversion 


    Assume for the sake of simplicity that $z<0$ at all times. Show that given the values of $h$, $\dot{h}$, $\ddot{h}$ and $h^{(3)}$, one can uniquely compute the booster state (the values of $x$, $\dot{x}$, $y$, $\dot{y}$, $\theta$, $\dot{\theta}$) and auxiliary system state (the values of $z$ and $\dot{z}$).

    Implement the corresponding function `T_inv`.
    """
    )
    return


@app.cell
def _(np):
    def T_inv(h_x_in, h_y_in, dh_x_in, dh_y_in,
              d2h_x_in, d2h_y_in, d3h_x_in, d3h_y_in,
              l, M, g):
        A_hddot = d2h_x_in
        B_hddot = d2h_y_in + g
        theta_out = np.arctan2(A_hddot, -B_hddot)
   
        z_over_M_squared = A_hddot**2 + B_hddot**2

        if z_over_M_squared < 1e-18: 
        
            z_out = -1e-9 * M 
        else:
            z_over_M = -np.sqrt(z_over_M_squared)
            z_out = M * z_over_M

        if z_out >= -1e-9: 

            if z_out > 1e-9: 

                pass 
            elif z_out > -1e-9 : 
                z_out = -1e-9 * M


        sin_theta_out = np.sin(theta_out)
        cos_theta_out = np.cos(theta_out)
        rhs_d3h_term1 = M * d3h_x_in
        rhs_d3h_term2 = M * d3h_y_in
        dtheta_times_z = cos_theta_out * rhs_d3h_term1 + sin_theta_out * rhs_d3h_term2
        dz_out = sin_theta_out * rhs_d3h_term1 - cos_theta_out * rhs_d3h_term2
   
        if abs(z_out) < 1e-9:
            dtheta_out = 0.0 
        else:
            dtheta_out = dtheta_times_z / z_out

        x_out = h_x_in + (l / 3) * sin_theta_out
        y_out = h_y_in - (l / 3) * cos_theta_out
        dx_out = dh_x_in + (l / 3) * dtheta_out * cos_theta_out
        dy_out = dh_y_in + (l / 3) * dtheta_out * sin_theta_out
   
        return x_out, dx_out, y_out, dy_out, theta_out, dtheta_out, z_out, dz_out
    return (T_inv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Admissible Path Computation

    Implement a function

    ```python
    def compute(
        x_0,
        dx_0,
        y_0,
        dy_0,
        theta_0,
        dtheta_0,
        z_0,
        dz_0,
        x_tf,
        dx_tf,
        y_tf,
        dy_tf,
        theta_tf,
        dtheta_tf,
        z_tf,
        dz_tf,
        tf,
    ):
        ...

    ```

    that returns a function `fun` such that `fun(t)` is a value of `x, dx, y, dy, theta, dtheta, z, dz, f, phi` at time `t` that match the initial and final values provided as arguments to `compute`.
    """
    )
    return


@app.cell
def _(T_inv, np):
    def calculate_h_derivatives(x_state, dx_state, y_state, dy_state,
                                 theta_state, dtheta_state,
                                 z_state, dz_state,
                                 l, M, g):
        sin_theta = np.sin(theta_state)
        cos_theta = np.cos(theta_state)
        h_x = x_state - (l / 3) * sin_theta
        h_y = y_state + (l / 3) * cos_theta
        dh_x = dx_state - (l / 3) * dtheta_state * cos_theta
        dh_y = dy_state - (l / 3) * dtheta_state * sin_theta
        d2h_x = (z_state / M) * sin_theta
        d2h_y = -(z_state / M) * cos_theta - g
        d3h_x = (1 / M) * (cos_theta * dtheta_state * z_state + sin_theta * dz_state)
        d3h_y = (1 / M) * (sin_theta * dtheta_state * z_state - cos_theta * dz_state)
        return h_x, h_y, dh_x, dh_y, d2h_x, d2h_y, d3h_x, d3h_y



    def get_poly_coeffs_deg7(y0, dy0, ddy0, dddy0, yf, dyf, ddyf, dddyf, tf_poly):
        if abs(tf_poly) < 1e-9: # tf est nul ou très petit
            if not (np.allclose(y0,yf) and np.allclose(dy0,dyf) and \
                    np.allclose(ddy0,ddyf) and np.allclose(dddy0,dddyf)):
                 return np.array([y0, dy0, ddy0/2, dddy0/6, 0, 0, 0, 0]) # Approximation
            return np.array([y0, dy0, ddy0/2, dddy0/6, 0, 0, 0, 0])

        M_poly = np.array([
            [1, 0,  0,   0,    0,     0,      0,       0     ],
            [0, 1,  0,   0,    0,     0,      0,       0     ],
            [0, 0,  2,   0,    0,     0,      0,       0     ],
            [0, 0,  0,   6,    0,     0,      0,       0     ],
            [1, tf_poly, tf_poly**2, tf_poly**3, tf_poly**4,  tf_poly**5,   tf_poly**6,    tf_poly**7  ],
            [0, 1, 2*tf_poly,3*tf_poly**2,4*tf_poly**3,5*tf_poly**4, 6*tf_poly**5,  7*tf_poly**6  ],
            [0, 0,  2,  6*tf_poly, 12*tf_poly**2,20*tf_poly**3,30*tf_poly**4, 42*tf_poly**5 ],
            [0, 0,  0,   6,  24*tf_poly, 60*tf_poly**2,120*tf_poly**3,210*tf_poly**4]
        ])
        Y_conditions = np.array([y0, dy0, ddy0, dddy0, yf, dyf, ddyf, dddyf])
        try:
            coeffs = np.linalg.solve(M_poly, Y_conditions)
        except np.linalg.LinAlgError:
            print("Erreur: La matrice polynomiale est singulière. Vérifiez tf et les conditions.")
            coeffs = np.zeros(8) # Retourner des zéros en cas d'erreur
            coeffs[0] = y0 # Au moins la condition initiale
        return coeffs

    def poly_val(coeffs, t_poly, order=0):
        val = 0.0
        if order == 0:
            for i in range(len(coeffs)): val += coeffs[i] * (t_poly**i)
        elif order == 1:
            for i in range(1, len(coeffs)): val += i * coeffs[i] * (t_poly**(i-1 if i > 1 else 0) if t_poly != 0 or i-1 >= 0 else (coeffs[1] if i==1 else 0.0) )
        elif order == 2:
            for i in range(2, len(coeffs)): val += i*(i-1) * coeffs[i] * (t_poly**(i-2 if i > 2 else 0) if t_poly != 0 or i-2 >= 0 else (2*coeffs[2] if i==2 else 0.0))
        elif order == 3:
            for i in range(3, len(coeffs)): val += i*(i-1)*(i-2) * coeffs[i] * (t_poly**(i-3 if i > 3 else 0) if t_poly != 0 or i-3 >= 0 else (6*coeffs[3] if i==3 else 0.0))
        elif order == 4:
            for i in range(4, len(coeffs)): val += i*(i-1)*(i-2)*(i-3) * coeffs[i] * (t_poly**(i-4 if i > 4 else 0) if t_poly != 0 or i-4 >= 0 else (24*coeffs[4] if i==4 else 0.0))
        else:
            raise ValueError("Ordre de dérivation non supporté pour poly_val")
        return val
    def compute(
        x_0_comp, dx_0_comp, y_0_comp, dy_0_comp, theta_0_comp, dtheta_0_comp, z_0_comp, dz_0_comp,
        x_tf_comp, dx_tf_comp, y_tf_comp, dy_tf_comp, theta_tf_comp, dtheta_tf_comp, z_tf_comp, dz_tf_comp,
        tf_comp,
        l_comp, M_comp, g_comp
    ):
        h_vals_0 = calculate_h_derivatives(x_0_comp, dx_0_comp, y_0_comp, dy_0_comp, theta_0_comp, dtheta_0_comp,
                                         z_0_comp, dz_0_comp, l_comp, M_comp, g_comp)
        h_vals_tf = calculate_h_derivatives(x_tf_comp, dx_tf_comp, y_tf_comp, dy_tf_comp, theta_tf_comp, dtheta_tf_comp,
                                          z_tf_comp, dz_tf_comp, l_comp, M_comp, g_comp)

        coeffs_hx = get_poly_coeffs_deg7(h_vals_0[0], h_vals_0[2], h_vals_0[4], h_vals_0[6],
                                         h_vals_tf[0], h_vals_tf[2], h_vals_tf[4], h_vals_tf[6], tf_comp)
        coeffs_hy = get_poly_coeffs_deg7(h_vals_0[1], h_vals_0[3], h_vals_0[5], h_vals_0[7],
                                         h_vals_tf[1], h_vals_tf[3], h_vals_tf[5], h_vals_tf[7], tf_comp)

        trajectory_params = {
            "coeffs_hx": coeffs_hx, "coeffs_hy": coeffs_hy,
            "l": l_comp, "M": M_comp, "g": g_comp
        }

        def fun(t_fun):
            params = trajectory_params
            c_hx, c_hy = params["coeffs_hx"], params["coeffs_hy"]
            l_f, M_f, g_f = params["l"], params["M"], params["g"]

            h_x_t    = poly_val(c_hx, t_fun, 0); h_y_t    = poly_val(c_hy, t_fun, 0)
            dh_x_t   = poly_val(c_hx, t_fun, 1); dh_y_t   = poly_val(c_hy, t_fun, 1)
            d2h_x_t  = poly_val(c_hx, t_fun, 2); d2h_y_t  = poly_val(c_hy, t_fun, 2)
            d3h_x_t  = poly_val(c_hx, t_fun, 3); d3h_y_t  = poly_val(c_hy, t_fun, 3)
            d4h_x_t  = poly_val(c_hx, t_fun, 4); d4h_y_t  = poly_val(c_hy, t_fun, 4) # u1, u2
       
            u_t = np.array([d4h_x_t, d4h_y_t])

            x_t, dx_t, y_t, dy_t, theta_t, dtheta_t, z_t, dz_t = \
                T_inv(h_x_t, h_y_t, dh_x_t, dh_y_t,
                      d2h_x_t, d2h_y_t, d3h_x_t, d3h_y_t,
                      l_f, M_f, g_f)
       
            if np.isnan(dtheta_t): 

                return x_t, dx_t, y_t, dy_t, theta_t, dtheta_t, z_t, dz_t, np.nan, np.nan


            sin_th_t, cos_th_t = np.sin(theta_t), np.cos(theta_t)
            R_pi2_minus_theta = np.array([[sin_th_t, cos_th_t], [-cos_th_t, sin_th_t]])
            M_R_u = M_f * (R_pi2_minus_theta @ u_t)
            additive_term_v = np.array([dtheta_t**2 * z_t, -2 * dtheta_t * dz_t])
            v_t = M_R_u + additive_term_v
            v1_t, v2_t = v_t[0], v_t[1]

            z_eff_t = z_t - M_f * (l_f/3) * dtheta_t**2
            F_rad_eff_t = (M_f * l_f * v2_t) / (3 * z_t) if abs(z_t) > 1e-9 else 0.0
           
            f_t = np.sqrt(z_eff_t**2 + F_rad_eff_t**2)
            phi_t = np.arctan2(-F_rad_eff_t, z_eff_t) 
       
            return x_t, dx_t, y_t, dy_t, theta_t, dtheta_t, z_t, dz_t, f_t, phi_t
        return fun
    return (compute,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Graphical Validation

    Test your `compute` function with

      - `x_0, dx_0, y_0, dy_0, theta_0, dtheta_0, z_0, dz_0 = 5.0, 0.0, 20.0, -1.0, -np.pi/8, 0.0, -M*g, 0.0`,
      - `x_tf, dx_tf, y_tf, dy_tf, theta_tf, dtheta_tf, z_tf, dz_tf = 0.0, 0.0, 4/3*l, 0.0,     0.0, 0.0, -M*g, 0.0`,
      - `tf = 10.0`.

    Make the graph of the relevant variables as a function of time, then make a video out of the same result. Comment and iterate if necessary!
    """
    )
    return


@app.cell
def _(compute, np, plt):
    from IPython.display import HTML, display
    import matplotlib.animation as animation
    import os
    import matplotlib.patches as patches


    l_param_glob = 1.0
    M_param_glob = 1.0
    g_param_glob = 1.0 
                   

    x_0_test, dx_0_test, y_0_test, dy_0_test = 5.0, 0.0, 20.0, -1.0
    theta_0_test, dtheta_0_test = -np.pi/8, 0.0
    z_0_test, dz_0_test = -M_param_glob*g_param_glob, 0.0

    x_tf_test, dx_tf_test, y_tf_test, dy_tf_test = 0.0, 0.0, (4/3)*l_param_glob, 0.0
    theta_tf_test, dtheta_tf_test = 0.0, 0.0
    z_tf_test, dz_tf_test = -M_param_glob*g_param_glob, 0.0

    tf_duration_test = 10.0

    print("--- Test de la fonction compute ---")
    trajectory_gen_func = compute(
        x_0_test, dx_0_test, y_0_test, dy_0_test, theta_0_test, dtheta_0_test, z_0_test, dz_0_test,
        x_tf_test, dx_tf_test, y_tf_test, dy_tf_test, theta_tf_test, dtheta_tf_test, z_tf_test, dz_tf_test,
        tf_duration_test,
        l_param_glob, M_param_glob, g_param_glob
    )

    times_for_eval = np.array([0.0, tf_duration_test / 2, tf_duration_test])
    print(f"\nTrajectoire de t=0 à t={tf_duration_test:.1f}s. Objectif x_tf={x_tf_test}, y_tf={y_tf_test:.2f}, theta_tf={theta_tf_test}")
    print("t   | x    | dx   | y    | dy   | theta | dtheta | z    | dz   | f    | phi (deg)")
    print("----|------|------|------|------|-------|--------|------|------|------|-----------")

    trajectory_data = []
    for t_val in times_for_eval:
        res_tuple = trajectory_gen_func(t_val)
        trajectory_data.append(res_tuple)
        print(f"{t_val:<4.1f}| {res_tuple[0]:<5.2f}| {res_tuple[1]:<5.2f}| {res_tuple[2]:<5.2f}| {res_tuple[3]:<5.2f}| "
              f"{res_tuple[4]:<6.2f}| {res_tuple[5]:<7.2f}| {res_tuple[6]:<5.2f}| {res_tuple[7]:<5.2f}| "
              f"{res_tuple[8]:<5.2f}| {np.rad2deg(res_tuple[9]):<7.1f}")


    def draw_booster_for_video(ax, x_center_mass, y_center_mass, theta_fus_rad,
                               f_poussee_mag, phi_flamme_rel_CoM_Base, # 
                               l_demi_longueur, M_masse, g_grav):
        artists = []
        BOOSTER_WIDTH = l_demi_longueur * 0.4
        FLAME_BASE_WIDTH = BOOSTER_WIDTH * 0.85
        MIN_THRUST_FOR_FLAME = 1e-3
        MAX_VISUAL_FLAME_LENGTH = l_demi_longueur * 3.5 


        u_CoM_vers_Nez = np.array([-np.sin(theta_fus_rad), np.cos(theta_fus_rad)])
        u_CoM_vers_Base = -u_CoM_vers_Nez 

        P_origine_flamme = np.array([x_center_mass, y_center_mass]) - l_demi_longueur * u_CoM_vers_Base
        u_fus_perp_droite = np.array([np.cos(theta_fus_rad), np.sin(theta_fus_rad)]) 
        P_nez_centre = np.array([x_center_mass, y_center_mass]) - l_demi_longueur * u_CoM_vers_Nez

        C1_nez_droite = P_nez_centre + (BOOSTER_WIDTH / 2) * u_fus_perp_droite
        C2_nez_gauche = P_nez_centre - (BOOSTER_WIDTH / 2) * u_fus_perp_droite
        C3_base_gauche = P_origine_flamme - (BOOSTER_WIDTH / 2) * u_fus_perp_droite
        C4_base_droite = P_origine_flamme + (BOOSTER_WIDTH / 2) * u_fus_perp_droite
   
        corps_fusée = patches.Polygon([C1_nez_droite, C2_nez_gauche, C3_base_gauche, C4_base_droite],
                                      closed=True, fc='darkgrey', ec='black', zorder=10)
        ax.add_patch(corps_fusée)
        artists.append(corps_fusée)

        if f_poussee_mag > MIN_THRUST_FOR_FLAME and not np.isnan(f_poussee_mag) and not np.isnan(phi_flamme_rel_CoM_Base):

       
            e_CoM_Base_vec = u_CoM_vers_Base
            e_CoM_Base_Perp_vec = np.array([np.cos(theta_fus_rad), np.sin(theta_fus_rad)])

            u_direction_flamme = (np.cos(phi_flamme_rel_CoM_Base) * e_CoM_Base_vec +
                                  np.sin(phi_flamme_rel_CoM_Base) * e_CoM_Base_Perp_vec)
       
            longueur_flamme = np.clip((l_demi_longueur / (M_masse * g_grav if M_masse * g_grav > 1e-6 else 1.0)) * f_poussee_mag,
                                      0, MAX_VISUAL_FLAME_LENGTH)
            if longueur_flamme < l_demi_longueur * 0.1 and f_poussee_mag > MIN_THRUST_FOR_FLAME: 
                 longueur_flamme = l_demi_longueur * 0.1

            P_sommet_flamme = P_origine_flamme + longueur_flamme * u_direction_flamme
       

            u_flamme_perp = np.array([-u_direction_flamme[1], u_direction_flamme[0]])
            P_flamme_gauche = P_origine_flamme - (FLAME_BASE_WIDTH / 2) * u_flamme_perp
            P_flamme_droite = P_origine_flamme + (FLAME_BASE_WIDTH / 2) * u_flamme_perp
       
            flamme_poly = patches.Polygon([P_flamme_gauche, P_flamme_droite, P_sommet_flamme],
                                          closed=True, fc='orangered', ec='none', alpha=0.75, zorder=5)
            ax.add_patch(flamme_poly)
            artists.append(flamme_poly)
        return artists

    def create_trajectory_video(nom_scenario, traj_func, t_fin_vid, fps_vid=25,
                                l_vid=l_param_glob, M_vid=M_param_glob, g_vid=g_param_glob):
        print(f"Génération de la vidéo pour : {nom_scenario}")
        fig_vid, ax_vid = plt.subplots(figsize=(8,10)) # Portrait pour voir y
   
        num_frames_vid = int(t_fin_vid * fps_vid)
        t_frames_vid = np.linspace(0, t_fin_vid, num_frames_vid)

        # Obtenir toutes les données pour déterminer les limites du graphe
        all_states_for_lims = np.array([traj_func(t) for t in t_frames_vid])
        x_coords_lim = all_states_for_lims[:,0]
        y_coords_lim = all_states_for_lims[:,2]

        ax_vid.set_xlim(min(x_coords_lim.min(), -l_vid*1), max(x_coords_lim.max(), x_0_test + l_vid*1) + l_vid*2)
        ax_vid.set_ylim(min(y_coords_lim.min(), -l_vid*0.5), max(y_coords_lim.max(), y_0_test + l_vid*1) + l_vid*1)
        ax_vid.set_aspect('equal', adjustable='box')
        ax_vid.grid(True, linestyle='--', alpha=0.7)
        ax_vid.set_xlabel("x (m)"); ax_vid.set_ylabel("y (m)")
        ax_vid.set_title(f"Trajectoire Fusée: {nom_scenario}")

        ground_line_vid = ax_vid.axhline(0, color='darkgreen', lw=2, zorder=1)
        target_landing_pad_vid = patches.Rectangle((x_tf_test - l_vid/2, y_tf_test - l_vid/10), l_vid, l_vid/5,
                                                facecolor='lightgreen', edgecolor='green', zorder=1.5)
        ax_vid.add_patch(target_landing_pad_vid)
        target_center_vid = ax_vid.plot(x_tf_test, y_tf_test, 'X', color='darkred', ms=10, zorder=1.6)[0]
   
        time_text_vid = ax_vid.text(0.02, 0.98, '', transform=ax_vid.transAxes, fontsize=9, va='top',
                                    bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.8))
        dynamic_artists_vid = []

        def animate_video_frame(i_frame):
            nonlocal dynamic_artists_vid
            for artist in dynamic_artists_vid: artist.remove()
            dynamic_artists_vid = []

            t_current_frame = t_frames_vid[i_frame]
            s = traj_func(t_current_frame) # x,dx,y,dy,th,dth,z,dz,f,phi
       

            new_artists = draw_booster_for_video(ax_vid, s[0],s[2],s[4], s[8],s[9],
                                                 l_vid, M_vid, g_vid)
            dynamic_artists_vid.extend(new_artists)
            time_text_vid.set_text(f"t={t_current_frame:.2f}s\n"
                                   f"x={s[0]:.1f},y={s[2]:.1f},θ={np.rad2deg(s[4]):.1f}°\n"
                                   f"f={s[8]:.1f},φ={np.rad2deg(s[9]):.1f}°")
            return dynamic_artists_vid + [time_text_vid]

        ani_vid = animation.FuncAnimation(fig_vid, animate_video_frame, frames=num_frames_vid,
                                          interval=1000/fps_vid, blit=False)
   
        video_dir_output = "videos"
        if not os.path.exists(video_dir_output): os.makedirs(video_dir_output)
        filename_vid = f"trajectory_{nom_scenario.lower().replace(' ', '_')}.mp4"
        full_path_vid = os.path.join(video_dir_output, filename_vid)

        try:
            ani_vid.save(full_path_vid, writer='ffmpeg', fps=fps_vid, dpi=150)
            print(f"Vidéo sauvegardée : {full_path_vid}")
            plt.close(fig_vid)
            return HTML(f"""<video width="480" height="600" controls autoplay loop><source src="{full_path_vid}" type="video/mp4"></video>""")
        except Exception as e_vid:
            print(f"Erreur sauvegarde vidéo pour {nom_scenario}: {e_vid}")
            plt.close(fig_vid)
            return None


    video_widget_test = create_trajectory_video(
        "Test1_Compute_Function",
        trajectory_gen_func,
        tf_duration_test,
        fps_vid=20 
    )
    if video_widget_test:
        display(video_widget_test)
    return


if __name__ == "__main__":
    app.run()
