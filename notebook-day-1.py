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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
def _():
    import scipy
    import scipy.integrate as sci

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import matplotlib.patches as patches

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, np, patches, plt, sci, tqdm


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


@app.cell
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return


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


@app.cell
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
    (mo.video(src=_filename))
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


@app.cell
def _():
    g = 1.0  
    M = 1.0  
    l = 1.0  
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
    Les composantes de la force appliquée au booster par le réacteur sont :

    $f_x = -f \cdot \sin(\theta + \phi)$

    $f_y = f \cdot \cos(\theta + \phi)$

    Où :

    $f$ : est la magnitude de la force 

    $\theta$ : est l'angle principal

    $\phi$ : est le déphasage

        Étapes de résolution :

    1. Dans le référentiel local du booster, la force du réacteur fait un angle $\phi$ avec l'axe du booster

    2. L'axe du booster lui-même fait un angle $\theta$ avec la verticale dans le référentiel global

    3. Par conséquent, dans le référentiel global, la force fait un angle $(\theta + \phi)$ avec la verticale

    4. La décomposition vectorielle de cette force donne alors :

       - Composante horizontale : $f_x = -f \cdot \sin(\theta + \phi)$

       - Composante verticale : $f_y = f \cdot \cos(\theta + \phi)$
    """
    )
    return


@app.cell
def _(np):
    f = 10.0

    def force_components(theta, phi):
        fx = f * - np.sin(theta + phi)
        fy = f * np.cos(theta + phi)
        return fx, fy

    theta = np.pi/6  
    phi = np.pi/4    

    fx, fy = force_components(theta, phi)
    fx, fy
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
    Les équations du mouvement pour le centre de masse sont :

    $\frac{d^2x}{dt^2} = \frac{f}{M} \cdot \sin(\theta + \phi)$

    $\frac{d^2y}{dt^2} = \frac{f}{M} \cdot \cos(\theta + \phi) - g$

    Où :

    $M$ : est la masse du booster

    $g$ : est la constante de gravité

    Étapes de résolution :

    1. Appliquons la seconde loi de Newton ($\vec{F} = m\vec{a}$) au centre de masse du booster

    2. Les forces agissant sur le booster sont :
       - La gravité : $(0, -Mg)$
       - La force du réacteur : $(f_x, f_y) = (f \cdot \sin(\theta + \phi), f \cdot \cos(\theta + \phi))$

    3. Selon l'axe $x$ :
       $M \cdot \frac{d^2x}{dt^2} = f_x = f \cdot \sin(\theta + \phi)$

       Donc $\frac{d^2x}{dt^2} = \frac{f}{M} \cdot \sin(\theta + \phi)$

    4. Selon l'axe $y$ :
       $M \cdot \frac{d^2y}{dt^2} = f_y - Mg = f \cdot \cos(\theta + \phi) - Mg$

       Donc $\frac{d^2y}{dt^2} = \frac{f}{M} \cdot \cos(\theta + \phi) - g$
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
    J = M * l**2 / 3
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
    L'équation qui gouverne l'angle d'inclinaison $\theta$ est :

       $\frac{d^2\theta}{dt^2} = \frac{l \cdot f \cdot \sin(\phi)}{J}$

       Étapes de résolution :

    Le moment cinétique est donné par : $\tau = J \cdot \frac{d^2\theta}{dt^2}$

    Le moment de force (couple) généré par le réacteur est :

    Force du réacteur : $f$

    Bras de levier : $l$ (distance entre le centre de masse et le réacteur)

    Angle entre la force et l'axe du booster : $\phi$

    La composante perpendiculaire à l'axe du booster est $f \cdot \sin(\phi)$

    Le couple résultant est donc :
    $\tau = l \cdot f \cdot \sin(\phi)$

    En égalisant les deux expressions du moment :
    $J \cdot \frac{d^2\theta}{dt^2} = l \cdot f \cdot \sin(\phi)$

    D'où :
    $\frac{d^2\theta}{dt^2} = \frac{l \cdot f \cdot \sin(\phi)}{J}$

    Avec $J = \frac{M \cdot l^2}{3}$ (moment d'inertie d'une tige uniforme), on obtient :
    $\frac{d^2\theta}{dt^2} = \frac{3 \cdot f \cdot \sin(\phi)}{M \cdot l}$

    En remplaçant les valeurs $M = 1$ et $l = 1$, on obtient la formule simplifiée :$\frac{d^2\theta}{dt^2} = 3 \cdot f \cdot \sin(\phi)$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
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


@app.cell
def _(J, M, g, l, np, plt, sci):
    def redstart_dynamics(t, state_vector, f_phi_func, M_const, g_const, l_const, J_const):

        x, dx, y, dy, theta, dtheta = state_vector
        f, phi = f_phi_func(t, state_vector)

        x_ddot = -(f / M_const) * np.sin(theta + phi)
        y_ddot = -g_const + (f / M_const) * np.cos(theta + phi)
        theta_ddot = (l_const * f / J_const) * np.sin(phi)


        derivatives = np.array([
            dx,
            x_ddot,
            dy,
            y_ddot,
            dtheta,
            theta_ddot
        ])
        return derivatives

    def redstart_solve(t_span, y0, f_phi_func):
    
        sol_scipy = sci.solve_ivp(
            fun=redstart_dynamics,
            t_span=t_span,
            y0=y0,
            args=(f_phi_func, M, g, l, J), 
            dense_output=True,             
            method='RK45'                  

        )

        return sol_scipy.sol

    # TEST 
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

        def f_phi_free_fall(t, y_state):
            return np.array([0.0, 0.0]) 

        print(f"\nRunning free_fall_example with y0={y0}...")
        sol_function = redstart_solve(t_span, y0, f_phi_free_fall)


        print(f"State at t=0: {sol_function(0.0)}")
        print(f"State at t=1.0: {sol_function(1.0)}")

        t_eval = np.linspace(t_span[0], t_span[1], 100)
        y_trajectory = sol_function(t_eval)
        x_t = y_trajectory[0]
        dx_t = y_trajectory[1]
        y_t = y_trajectory[2]
        dy_t = y_trajectory[3]
        theta_t = y_trajectory[4]
        dtheta_t = y_trajectory[5]

        plt.figure(figsize=(10, 6))
        plt.plot(t_eval, y_t, label=r"$y(t)$ (height of CoM in meters)")
        plt.plot(t_eval, l * np.ones_like(t_eval), color="grey", ls="--", label=r"Booster half-length line $y=\ell$")
        plt.plot(t_eval, y_t - l * np.cos(theta_t), color="green", ls=":", label=r"Base height $y(t) - \ell \cos(\theta(t))$")
        plt.plot(t_eval, y_t + l * np.cos(theta_t), color="purple", ls=":", label=r"Nose height $y(t) + \ell \cos(\theta(t))$")


        plt.title("Free Fall Test for Redstart Booster")
        plt.xlabel("Time $t$ (seconds)")
        plt.ylabel("Position $y$ (meters)")
        plt.grid(True)
        plt.legend()
        plt.ylim(min(y_t.min() -1, -l-1) , y0[2] + 1) 
        y_analytical = y0[2] - 0.5 * g * t_eval**2
        plt.plot(t_eval, y_analytical, 'r.', markersize=2, label="Analytical $y(t)$ (free fall)")
        plt.legend()
        return plt.gcf()


    fig = free_fall_example()
    plt.show()
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""la poussée f est nulle, le mouvement est uniquement gouverné par la gravité et les conditions initiales (hauteur de départ de 10m, vitesse initiale nulle), alors les courbes dans la figure sont y(t) (courbe bleue principale) : Altitude du centre de masse (la variable la plus directe issue de la simulation de la dynamique verticale de la fusée. Elle montre comment la hauteur du centre de gravité évolue avec le temps. C'est la courbe fondamentale) tandis que y_base(t) (courbe verte en pointillés) et y_nose(t) (courbe violette en pointillés) sont les altitudes de la base et du nez de la fusée.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0)$, can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell
def _(l, np, plt, redstart_solve):
    #define the derived f(t)
    def f_control_landing(t):
        return (108/125) * t - (29/25)

    def f_phi_controlled_landing(t, state_vector):
        f_val = f_control_landing(t)
        phi_val = 0.0  
        return np.array([f_val, phi_val])

    # Initial conditions 
    y0_landing = np.array([
        0.0,  # x
        0.0,  # dx/dt
        10.0, # y
        0.0,  # dy/dt
        0.0,  # theta
        0.0   # dtheta/dt
    ])

    t_final_landing = 5.0
    t_span_landing = [0.0, t_final_landing]

    print(f"Simulating controlled landing...")
    print(f"Initial state: {y0_landing}")
    print(f"Target y({t_final_landing}) = {l}, dy/dt({t_final_landing}) = 0")

    sol_landing_func = redstart_solve(t_span_landing, y0_landing, f_phi_controlled_landing)

    # Evaluate solution
    t_eval_landing = np.linspace(t_span_landing[0], t_span_landing[1], 500)
    trajectory_landing = sol_landing_func(t_eval_landing)

    x_land, dx_land, y_land, dy_land, theta_land, dtheta_land = trajectory_landing

    # Check final conditions
    print(f"\nSimulation results at t = {t_final_landing}:")
    print(f"  x({t_final_landing})     = {x_land[-1]:.4f} (Target: 0)")
    print(f"  dx/dt({t_final_landing}) = {dx_land[-1]:.4f} (Target: 0)")
    print(f"  y({t_final_landing})     = {y_land[-1]:.4f} (Target: {l})")
    print(f"  dy/dt({t_final_landing}) = {dy_land[-1]:.4f} (Target: 0)")
    print(f"  theta({t_final_landing}) = {np.degrees(theta_land[-1]):.4f} deg (Target: 0)")
    print(f"  dtheta/dt({t_final_landing}) = {np.degrees(dtheta_land[-1]):.4f} deg/s (Target: 0)")

    fig_land, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(t_eval_landing, y_land, label=r'$y(t)$ (CoM height)')
    axs[0].plot(t_eval_landing, y_land - l * np.cos(theta_land), ls="--", label=r'$y_{base}(t)$')
    axs[0].axhline(l, color='r', ls=':', label=f'Target $y={l}$')
    axs[0].axhline(0, color='k', ls='-', lw=0.5)
    axs[0].set_ylabel('Position y (m)')
    axs[0].legend()
    axs[0].grid(True)

    ax0_twin = axs[0].twinx()
    ax0_twin.plot(t_eval_landing, dy_land, color='green', label=r'$\dot{y}(t)$ (CoM velocity)')
    ax0_twin.axhline(0, color='green', ls=':', label=r'Target $\dot{y}=0$')
    ax0_twin.set_ylabel(r'Velocity $\dot{y}$ (m/s)', color='green')
    ax0_twin.tick_params(axis='y', labelcolor='green')
    ax0_twin.legend(loc='center right')

    axs[0].set_title('Controlled Landing: Vertical Motion')

    # Plot theta and dtheta/dt (should be zero)
    axs[1].plot(t_eval_landing, np.degrees(theta_land), label=r'$\theta(t)$ (Booster angle)')
    axs[1].set_ylabel('Angle (degrees)')
    axs[1].legend()
    axs[1].grid(True)

    ax1_twin = axs[1].twinx()
    ax1_twin.plot(t_eval_landing, np.degrees(dtheta_land), color='purple', label=r'$\dot{\theta}(t)$ (Angular velocity)')
    ax1_twin.set_ylabel('Angular Velocity (deg/s)', color='purple')
    ax1_twin.tick_params(axis='y', labelcolor='purple')
    ax1_twin.legend(loc='center right')
    axs[1].set_title('Rotational Motion (Should be Zero)')


    f_values_eval = np.array([f_phi_controlled_landing(t, None)[0] for t in t_eval_landing])
    phi_values_eval = np.array([f_phi_controlled_landing(t, None)[1] for t in t_eval_landing])

    axs[2].plot(t_eval_landing, f_values_eval, label=r'$f(t)$ (Thrust Magnitude)')
    axs[2].axhline(0, color='k', ls=':', label='f=0 line')
    axs[2].set_ylabel('Force f (N)')
    axs[2].legend(loc='upper left')
    axs[2].grid(True)

    ax2_twin = axs[2].twinx()
    ax2_twin.plot(t_eval_landing, np.degrees(phi_values_eval), color='orange', label=r'$\phi(t)$ (Thrust Angle)')
    ax2_twin.set_ylabel(r'Angle $\phi$ (degrees)', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2_twin.legend(loc='center right')

    axs[2].set_title('Control Inputs')
    axs[2].set_xlabel('Time t (s)')

    plt.tight_layout()
    plt.show()

    f_at_t0 = f_control_landing(0)
    print(f"\nCalculated f(0) = {f_at_t0:.4f} N")
    return


@app.cell
def _():
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


@app.cell
def _(np, patches, plt):
    def draw_booster_state(ax, x, y, theta, f_thrust, phi_gimbal,
                           booster_l, M_booster, g_gravity):
   
        BOOSTER_WIDTH = booster_l * 0.4  
        FLAME_BASE_WIDTH = BOOSTER_WIDTH * 0.85
        MIN_THRUST_FOR_FLAME = 1e-3
        MAX_VISUAL_FLAME_LENGTH = booster_l * 4.0

        u_axis_CoM_to_Nose = np.array([-np.sin(theta), np.cos(theta)])
        u_perp_CoM_to_Right = np.array([np.cos(theta), np.sin(theta)])

        C1_NoseRight = (np.array([x, y]) + booster_l * u_axis_CoM_to_Nose +
                        (BOOSTER_WIDTH / 2) * u_perp_CoM_to_Right)
        C2_NoseLeft = (np.array([x, y]) + booster_l * u_axis_CoM_to_Nose -
                       (BOOSTER_WIDTH / 2) * u_perp_CoM_to_Right)
        C3_BaseLeft = (np.array([x, y]) - booster_l * u_axis_CoM_to_Nose -
                       (BOOSTER_WIDTH / 2) * u_perp_CoM_to_Right)
        C4_BaseRight = (np.array([x, y]) - booster_l * u_axis_CoM_to_Nose +
                        (BOOSTER_WIDTH / 2) * u_perp_CoM_to_Right)
   
        booster_patch = patches.Polygon([C1_NoseRight, C2_NoseLeft, C3_BaseLeft, C4_BaseRight],
                                        closed=True, fc='slategrey', ec='black', zorder=10)
        ax.add_patch(booster_patch)

        if f_thrust > MIN_THRUST_FOR_FLAME:
            P_base_center = np.array([x, y]) - booster_l * u_axis_CoM_to_Nose

            e_theta_CoM_to_base = np.array([np.sin(theta), -np.cos(theta)])
            e_theta_perp_CCW = np.array([np.cos(theta), np.sin(theta)])

            u_flame_axis = (np.cos(phi_gimbal) * e_theta_CoM_to_base +
                            np.sin(phi_gimbal) * e_theta_perp_CCW)
       
            norm_flame_axis = np.linalg.norm(u_flame_axis)
            if norm_flame_axis > 1e-6:
                u_flame_axis /= norm_flame_axis
       
            if M_booster > 1e-6 and g_gravity > 1e-6:
                flame_L = (booster_l / (M_booster * g_gravity)) * f_thrust
            else:
                flame_L = booster_l * (f_thrust / (1.0 if M_booster < 1e-6 else M_booster))
       
            flame_L = np.clip(flame_L, 0, MAX_VISUAL_FLAME_LENGTH)
       
            u_flame_perp = np.array([-u_flame_axis[1], u_flame_axis[0]])

            P_flame_Apex = P_base_center + flame_L * u_flame_axis
            P_flame_BaseLeft = P_base_center - (FLAME_BASE_WIDTH / 2) * u_flame_perp
            P_flame_BaseRight = P_base_center + (FLAME_BASE_WIDTH / 2) * u_flame_perp
       
            flame_patch = patches.Polygon([P_flame_BaseLeft, P_flame_BaseRight, P_flame_Apex],
                                          closed=True, fc='orangered', ec='none', alpha=0.75, zorder=5)
            ax.add_patch(flame_patch)
   
        ground_exists = any(line.get_label() == '_ground_line' for line in ax.lines)
        if not ground_exists:
            ax.axhline(0, color='darkgreen', linewidth=2, label='_ground_line', zorder=1)

        target_exists = any(line.get_label() == '_target_marker' for line in ax.lines)
        if not target_exists:
            ax.plot(0, 0, 'X', color='red', markersize=15, markeredgewidth=3, label='_target_marker', zorder=2)

    fig2, ax = plt.subplots(figsize=(8, 10))

    BOOSTER_L_HALF = 1.0  
    BOOSTER_MASS = 1.0  
    GRAVITY_G = 1.0      

    x_com1 = 1.0 * BOOSTER_L_HALF
    y_com1 = 3.0 * BOOSTER_L_HALF
    theta_angle1_deg = 15.0
    theta1_rad = np.deg2rad(theta_angle1_deg)

    f1_thrust = BOOSTER_MASS * GRAVITY_G
    phi1_gimbal_rad = -theta1_rad

    print(f"Drawing Scenario 1: Tilted, f=Mg (Flame length should be {BOOSTER_L_HALF:.2f})")
    draw_booster_state(ax, x=x_com1, y=y_com1, theta=theta1_rad,
                           f_thrust=f1_thrust, phi_gimbal=phi1_gimbal_rad,
                           booster_l=BOOSTER_L_HALF, M_booster=BOOSTER_MASS, g_gravity=GRAVITY_G)
    ax.arrow(x_com1, y_com1, 0, 0.5 * BOOSTER_L_HALF, head_width=0.1, fc='blue', ec='blue', zorder=20)

    x_com2 = -2.0 * BOOSTER_L_HALF
    y_com2 = 4.0 * BOOSTER_L_HALF
    theta2_rad = np.deg2rad(0.0)

    f2_thrust = 2.0 * BOOSTER_MASS * GRAVITY_G
    phi2_gimbal_rad = np.deg2rad(0.0)

    print(f"\nDrawing Scenario 2: Vertical, f=2Mg (Flame length should be {2*BOOSTER_L_HALF:.2f})")
    draw_booster_state(ax, x=x_com2, y=y_com2, theta=theta2_rad,
                           f_thrust=f2_thrust, phi_gimbal=phi2_gimbal_rad,
                           booster_l=BOOSTER_L_HALF, M_booster=BOOSTER_MASS, g_gravity=GRAVITY_G)

    ax.arrow(x_com2, y_com2, 0, 0.5 * BOOSTER_L_HALF, head_width=0.1, fc='green', ec='green', zorder=20)


    x_com3 = -4.0 * BOOSTER_L_HALF
    y_com3 = 2.5 * BOOSTER_L_HALF
    theta3_rad = np.deg2rad(0.0)
   
    f3_thrust = 0.5 * BOOSTER_MASS * GRAVITY_G
    phi3_gimbal_rad = np.deg2rad(0.0)

    print(f"\nDrawing Scenario 3: Vertical, f=0.5Mg (Flame length should be {0.5*BOOSTER_L_HALF:.2f})")
    draw_booster_state(ax, x=x_com3, y=y_com3, theta=theta3_rad,
                           f_thrust=f3_thrust, phi_gimbal=phi3_gimbal_rad,
                           booster_l=BOOSTER_L_HALF, M_booster=BOOSTER_MASS, g_gravity=GRAVITY_G)
    ax.arrow(x_com3, y_com3, 0, 0.5 * BOOSTER_L_HALF, head_width=0.1, fc='purple', ec='purple', zorder=20)


    ax.set_title(f"Redstart Booster Visualization (ℓ={BOOSTER_L_HALF}, M={BOOSTER_MASS}, g={GRAVITY_G})")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
   
    ax.set_xlim(-6 * BOOSTER_L_HALF, 4 * BOOSTER_L_HALF)
    ax.set_ylim(-1 * BOOSTER_L_HALF, 7 * BOOSTER_L_HALF)
   
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.7)
       
    plt.show()
    return BOOSTER_L_HALF, BOOSTER_MASS


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualization

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell
def _(BOOSTER_L_HALF, BOOSTER_MASS, M, g, l, np, patches, plt):
    from scipy.integrate import solve_ivp
    from IPython.display import HTML, display
    import os 
    import matplotlib.animation as animation


    I_booster = (1/3) * BOOSTER_MASS * l**2
    def redstart_solve1(t_span, y0, f_phi_func, M_sim=M, g_sim=g, I_sim=I_booster, l_param_sim=l):
        def dynamics(t, y):
            x_dyn, dx_dyn, y_pos_dyn, dy_dyn, theta_dyn, dtheta_dyn = y
            f_thrust_magnitude, phi_thrust_angle = f_phi_func(t, y)

            sin_theta_val = np.sin(theta_dyn)
            cos_theta_val = np.cos(theta_dyn)
            e_theta = np.array([sin_theta_val, -cos_theta_val])
            e_theta_perp = np.array([cos_theta_val, sin_theta_val])

            f_exhaust_vec = f_thrust_magnitude * (np.cos(phi_thrust_angle) * e_theta + np.sin(phi_thrust_angle) * e_theta_perp)
            F_thrust_on_rocket = -f_exhaust_vec
            F_net = F_thrust_on_rocket + np.array([0.0, -M_sim * g_sim])

            ddx_dyn = F_net[0] / M_sim
            ddy_dyn = F_net[1] / M_sim
       
            torque = -l_param_sim * f_thrust_magnitude * np.sin(phi_thrust_angle)
            ddtheta_dyn = torque / I_sim

            return [dx_dyn, ddx_dyn, dy_dyn, ddy_dyn, dtheta_dyn, ddtheta_dyn]

        sol_ivp = solve_ivp(dynamics, t_span, y0, dense_output=True, rtol=1e-7, atol=1e-9)

        def sol(t):
            t_eval = np.atleast_1d(t)
            return sol_ivp.sol(t_eval).T
        return sol

    # === Drawing function (draw_booster_state) ===
    # (This function remains unchanged from your provided code)
    def draw_booster_state1(ax, x_draw, y_draw, theta_draw, f_thrust_draw, phi_gimbal_draw,
                           booster_l_draw=BOOSTER_L_HALF, M_booster_draw=BOOSTER_MASS, g_gravity_draw=g):
        artists = []
        BOOSTER_WIDTH = booster_l_draw * 0.4
        FLAME_BASE_WIDTH = BOOSTER_WIDTH * 0.85
        MIN_THRUST_FOR_FLAME = 1e-3
        MAX_VISUAL_FLAME_LENGTH = booster_l_draw * 4.0

        u_axis_CoM_to_Nose = np.array([-np.sin(theta_draw), np.cos(theta_draw)])
        u_perp_CoM_to_Right = np.array([np.cos(theta_draw), np.sin(theta_draw)])

        C1 = (np.array([x_draw, y_draw]) + booster_l_draw * u_axis_CoM_to_Nose + (BOOSTER_WIDTH / 2) * u_perp_CoM_to_Right)
        C2 = (np.array([x_draw, y_draw]) + booster_l_draw * u_axis_CoM_to_Nose - (BOOSTER_WIDTH / 2) * u_perp_CoM_to_Right)
        C3 = (np.array([x_draw, y_draw]) - booster_l_draw * u_axis_CoM_to_Nose - (BOOSTER_WIDTH / 2) * u_perp_CoM_to_Right)
        C4 = (np.array([x_draw, y_draw]) - booster_l_draw * u_axis_CoM_to_Nose + (BOOSTER_WIDTH / 2) * u_perp_CoM_to_Right)
   
        booster_patch = patches.Polygon([C1, C2, C3, C4], closed=True, fc='slategrey', ec='black', zorder=10)
        ax.add_patch(booster_patch)
        artists.append(booster_patch)

        if f_thrust_draw > MIN_THRUST_FOR_FLAME: # Only draw flame if thrust is positive
            P_base_center = np.array([x_draw, y_draw]) - booster_l_draw * u_axis_CoM_to_Nose
            e_theta_CoM_to_base = np.array([np.sin(theta_draw), -np.cos(theta_draw)])
            e_theta_perp_CCW = np.array([np.cos(theta_draw), np.sin(theta_draw)])
            u_flame_axis = (np.cos(phi_gimbal_draw) * e_theta_CoM_to_base + np.sin(phi_gimbal_draw) * e_theta_perp_CCW)
       
            norm_flame_axis = np.linalg.norm(u_flame_axis)
            if norm_flame_axis > 1e-6: u_flame_axis /= norm_flame_axis
       
            if M_booster_draw > 1e-6 and g_gravity_draw > 1e-6:
                flame_L = (booster_l_draw / (M_booster_draw * g_gravity_draw)) * f_thrust_draw
            else:
                flame_L = booster_l_draw * (f_thrust_draw / (1.0 if M_booster_draw < 1e-6 else M_booster_draw))
       
            flame_L = np.clip(flame_L, 0, MAX_VISUAL_FLAME_LENGTH)
       
            u_flame_perp = np.array([-u_flame_axis[1], u_flame_axis[0]])
            P_flame_Apex = P_base_center + flame_L * u_flame_axis
            P_flame_BaseLeft = P_base_center - (FLAME_BASE_WIDTH / 2) * u_flame_perp
            P_flame_BaseRight = P_base_center + (FLAME_BASE_WIDTH / 2) * u_flame_perp
       
            flame_patch = patches.Polygon([P_flame_BaseLeft, P_flame_BaseRight, P_flame_Apex],
                                          closed=True, fc='orangered', ec='none', alpha=0.75, zorder=5)
            ax.add_patch(flame_patch)
            artists.append(flame_patch)
   
        return artists

    # === Generic Animation Function ===
    def create_simulation_video(scenario_name, y0_anim, f_phi_controller_func,
                                t_duration=5.0, fps=25, video_filename_base="simulation"):
        print(f"Generating animation for: {scenario_name}")
   
        t_span_anim = [0.0, t_duration]
        sol_function = redstart_solve1(t_span_anim, y0_anim, f_phi_controller_func)

        num_frames = int(t_duration * fps)
        t_frames = np.linspace(t_span_anim[0], t_span_anim[1], num_frames)
   
        sim_states = sol_function(t_frames)
        sim_controls = np.array([f_phi_controller_func(t, sim_states[i,:]) for i, t in enumerate(t_frames)])

        fig, ax = plt.subplots(figsize=(8, 8)) # Square aspect ratio often good for general motion
   
        # Dynamic plot limits based on trajectory
        all_x_coords = sim_states[:, 0]
        all_y_coords = sim_states[:, 2]
        x_min = min(all_x_coords.min(), -1*BOOSTER_L_HALF) - 2*BOOSTER_L_HALF
        x_max = max(all_x_coords.max(), 1*BOOSTER_L_HALF) + 2*BOOSTER_L_HALF
        y_min = min(all_y_coords.min(), -1*BOOSTER_L_HALF) - 1*BOOSTER_L_HALF
        y_max = max(all_y_coords.max(), y0_anim[2] + 1*BOOSTER_L_HALF) # Ensure initial height is visible
   
        # Ensure a minimum view range if motion is small
        if (x_max - x_min) < 5 * BOOSTER_L_HALF:
            mid_x = (x_max + x_min) / 2
            x_min = mid_x - 2.5 * BOOSTER_L_HALF
            x_max = mid_x + 2.5 * BOOSTER_L_HALF
        if (y_max - y_min) < 12 * BOOSTER_L_HALF: # To accommodate initial y=10
            mid_y = (y_max + y_min) / 2
            y_min = mid_y - 6 * BOOSTER_L_HALF
            y_max = mid_y + 6 * BOOSTER_L_HALF


        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
        ax.set_title(f"Booster Simulation: {scenario_name}")

        ground_line = ax.axhline(0, color='darkgreen', linewidth=2, zorder=1, label="Ground (y=0)")
        # Target marker at (0,0) for general scenarios
        target_marker = ax.plot(0, 0, 'X', color='red', markersize=12, markeredgewidth=1.5, zorder=2, label="Origin (0,0)")[0]
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9, va='top',
                            bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.8))
        path_x, path_y = [], []
        path_line, = ax.plot([], [], 'c--', lw=1.5, alpha=0.6, label="Trajectory")
        ax.legend(loc='lower left', fontsize='small')

        dynamic_artists = []

        def animate_frame(i):
            nonlocal dynamic_artists, path_x, path_y
            for artist in dynamic_artists: artist.remove()
            dynamic_artists = []

            current_state_frame = sim_states[i, :]
            x_sim, _, y_sim, dy_sim, theta_sim, dtheta_sim = current_state_frame
            f_sim, phi_sim = sim_controls[i, :]

            path_x.append(x_sim)
            path_y.append(y_sim)
            if len(path_x) > 200: # Limit path trace length
                 path_x.pop(0); path_y.pop(0)
            path_line.set_data(path_x, path_y)
       
            new_artists = draw_booster_state1(ax, x_sim, y_sim, theta_sim, f_sim, phi_sim)
            dynamic_artists.extend(new_artists)
       
            time_text.set_text(f"Time: {t_frames[i]:.2f}s\ny={y_sim:.2f}m, dy={dy_sim:.2f}m/s\nθ={np.rad2deg(theta_sim):.1f}°, dθ={np.rad2deg(dtheta_sim):.1f}°/s\nF={f_sim:.2f}N, φ={np.rad2deg(phi_sim):.1f}°")
       
            return dynamic_artists + [time_text, path_line]

        ani = animation.FuncAnimation(fig, animate_frame, frames=num_frames, interval=1000/fps, blit=False)

        video_dir = "videos"
        if not os.path.exists(video_dir): os.makedirs(video_dir)
        video_filename = f"{video_filename_base}_{scenario_name.lower().replace(' ', '_').replace('=', '').replace('/', 'div')}.mp4"
        full_video_path = os.path.join(video_dir, video_filename)

        try:
            ani.save(full_video_path, writer='ffmpeg', fps=fps, dpi=100) # Lower DPI for faster generation
            print(f"Animation saved as {full_video_path}")
            plt.close(fig)
            return HTML(f"""<video width="480" height="480" controls autoplay loop><source src="{full_video_path}" type="video/mp4">Your browser does not support the video tag.</video>""")
        except Exception as e:
            print(f"Error saving or displaying animation for {scenario_name}: {e}")
            plt.close(fig)
            return None

    # --- Define Scenarios and Run Animations ---
    if __name__ == "__main__":
        # Initial state for all three scenarios
        y0_common = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        simulation_duration = 5.0

        # Scenario 1: Free Fall (f=0, phi=0)
        def f_phi_s1(t, y_state):
            return np.array([0.0, 0.0])
        video_s1 = create_simulation_video("Free_Fall_f0_phi0", y0_common, f_phi_s1, t_duration=simulation_duration)
        if video_s1: display(video_s1)

        # Scenario 2: Hover (f=Mg, phi=0)
        def f_phi_s2(t, y_state):
            return np.array([M * g, 0.0])
        video_s2 = create_simulation_video("Hover_fMg_phi0", y0_common, f_phi_s2, t_duration=simulation_duration)
        if video_s2: display(video_s2)

        # Scenario 3: Gimbaled Hover (f=Mg, phi=pi/8)
        def f_phi_s3(t, y_state):
            return np.array([M * g, np.pi/8])
        video_s3 = create_simulation_video("Gimbaled_Hover_fMg_phipi_div_8", y0_common, f_phi_s3, t_duration=simulation_duration)
        if video_s3: display(video_s3)

    return


if __name__ == "__main__":
    app.run()
