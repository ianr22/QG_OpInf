import numpy as np
from scipy.sparse import spdiags, lil_matrix
import scipy.sparse as sp
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def diff(n, dx):
    """
    Computes the finite difference operators

    Parameters
    ----------
    n : int
        size of the matrix
    dx : float
        discretization in space

    Returns
    ----------
    Dxf : sparse matrix
        Forward difference operator.
    Dx : sparse matrix
        Centered first derivative
    Lx : sparse matrix
        Second derivative (Laplacian)
    D3x : sparse matrix
        Third derivative
    """
    # Forward difference operator Dxf
    val = 1 / dx
    e1 = val * np.ones(n)
    Dxf = spdiags([-e1, e1], [0, 1], n, n, format='lil')
    Dxf[-1, 0] = val

    # Centered first derivative Dx
    val = 0.5 / dx
    e1 = val * np.ones(n)
    Dx = spdiags([-e1, e1], [-1, 1], n, n, format='lil')
    Dx[0, -1] = -val
    Dx[-1, 0] = val

    # Second derivative (Laplacian) Lx
    val = 1 / dx**2
    e1 = val * np.ones(n)
    Lx = spdiags([e1, -2*e1, e1], [-1, 0, 1], n, n, format='lil')
    Lx[0, -1] = val
    Lx[-1, 0] = val

    # Third derivative D3x
    val2 = 1 / dx**3
    e1 = val2 * np.ones(n)
    D3x = spdiags(
        [-0.5 * e1, e1, -e1, 0.5 * e1],
        [-2, -1, 1, 2],
        n, n, format='lil'
    )
    # Periodic BCs
    D3x[0, -2] = -0.5 * val2
    D3x[0, -1] = val2
    D3x[-1, 0] = -val2
    D3x[-1, 1] = 0.5 * val2
    D3x[1, -1] = -0.5 * val2
    D3x[-2, 0] = 0.5 * val2

    return Dxf.tocsr(), Dx.tocsr(), Lx.tocsr(), D3x.tocsr()


def rhs(t, y, nu, alp, rho, Dx, D3x):
    """
    Computes the right-hand side of the KdV-type PDE.

    Parameters
    ----------
    y : ndarray
        State vector (1D array).
    nu : float
        Dispersion coefficient.
    alp : float
        Nonlinear coefficient.
    rho : float
        Advection coefficient.
    Dx : sparse matrix
        First derivative matrix.
    D3x : sparse matrix
        Third derivative matrix.

    Returns
    -------
    dy : ndarray
        Time derivative dy/dt (rhs)
    """
    y_x = Dx @ y
    dy = alp * (Dx @ (y**2)) + rho * y_x + nu * (D3x @ y)
    return dy


def get_rom_operators(Phi, D1, D3, r):
    """
    Computes reduced-order operators for ROM of KdV-type PDE with sparse differential operators.

    Parameters
    ----------
    Phi : ndarray
        Basis matrix (n x r), dense.
    D1 : scipy.sparse matrix
        First-derivative operator (n x n), sparse.
    D3 : scipy.sparse matrix
        Third-derivative operator (n x n), sparse.
    r : int
        Number of modes in the reduced basis.

    Returns
    -------
    rl3 : ndarray
        ROM linear operator for third derivative (r x r).
    rl1 : ndarray
        ROM linear operator for first derivative (r x r).
    rq : ndarray
        ROM nonlinear operator (r x r x r), permuted to match MATLAB output.
    """

    # Linear ROM operators
    rl3 = Phi.T @ (D3 @ Phi) if sp.issparse(D3) else Phi.T @ D3 @ Phi
    rl1 = Phi.T @ (D1 @ Phi) if sp.issparse(D1) else Phi.T @ D1 @ Phi

    # Nonlinear ROM operator
    rq = np.zeros((r, r, r))
    D1Phi = D1.dot(Phi) if sp.issparse(D1) else D1 @ Phi  # (n x r)
    # D1Phi = D1 @ Phi

    # TODO: try einsum
    for i in range(r):
        for j in range(r):
            phi_j = Phi[:, j]
            for k in range(r):
                phi_jDphi_k = phi_j * D1Phi[:, k]
                rq[i, j, k] = 2 * (Phi[:, i].T @ phi_jDphi_k)

    rq = np.transpose(rq, (1, 2, 0))

    return rl3, rl1, rq

def rhsrom(t, y, alp, nu, rho, L3, L1, Q):
    """
    Computes the reduced-order model RHS for the KdV equation.

    Parameters
    ----------
    y : ndarray
        Reduced state vector (r,)
    alp : float
        Coefficient for the quadratic term
    nu : float
        Coefficient for third derivative term
    rho : float
        Coefficient for first derivative term
    L3 : ndarray
        ROM linear operator for third derivative (r x r)
    L1 : ndarray
        ROM linear operator for first derivative (r x r)
    Q : ndarray
        ROM nonlinear operator (r x r x r), Q[i,j,k] = contribution to i-th mode

    Returns
    -------
    dyrom : ndarray
        Time derivative of reduced state (r,)
    """
    r = y.shape[0]
    yyt = np.outer(y, y)  # (r x r)
    quad = np.array([np.sum(Q[:, :, i] * yyt) for i in range(r)])

    # Linear + nonlinear combination
    dyrom = (nu * L3 + rho * L1) @ y + alp * quad
    return dyrom

def plot_trajectory(x_plot, x_value, dt, title, name_of_file):
    """
    Parameters
    -----------
    x_plot : ndarray
    x_value : ndarray
    """
    fig, ax = plt.subplots()
    line, = ax.plot(x_plot, x_value[:, 0], lw=2)

    ax.set_xlim([-10, 10])
    ax.set_ylim([-1, 9])
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(title)

    def update(frame):
        line.set_ydata(x_value[:, frame])
        ax.set_title(f't = {frame * dt:.2f}')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(x_value.shape[1]), blit=True, interval=20)
    plt.show()
    ani.save(name_of_file, writer='pillow')

def animate_multiple_trajectories(x_plot, true_val, mean_val, dt, title, name_of_file):
    """

    :param x_plot:
    :param true_val:
    :param mean_val:
    :param dt:
    :param title:
    :param name_of_file:
    :return:
    """
    fig, ax = plt.subplots()
    line1, = ax.plot(x_plot, true_val[:, 0], lw=2)
    line2, = ax.plot(x_plot, mean_val[:, 0], lw=2)

    ax.set_xlim([-10, 10])
    ax.set_ylim([-1, 9])
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(title)

    def update(frame):
        line1.set_ydata(true_val[:, frame])
        line2.set_ydata(mean_val[:, frame])
        ax.set_title(f't = {frame * dt:.2f}')
        return line1, line2

    ani = animation.FuncAnimation(fig, update, frames=range(true_val.shape[1]), blit=True, interval=20)
    plt.show()
    ani.save(name_of_file, writer='pillow')

