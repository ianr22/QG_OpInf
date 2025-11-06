import numpy as np
import scipy.special as special
from scipy.sparse import spdiags, lil_matrix
import scipy.sparse as sp

def distribute_nx(rank, nx, size):
	"""
 	distribute_nx distributes the spatial DoF nx into chunks of size nxi such that 
 	\sum_{i=0}^{p-1} nx_i = nx where p is the number of used compute cores

 	:rank: 	the MPI rank 0, 1, ... ,p-1 that will run this function
 	:n_x: 	number of DoF used for spatial discretization
 	:size: 	size of the MPI communicator (p in our case)
 	
 	:return: the start and end index of the local DoF, and the number of local DoF for each rank
 	"""

	nx_i_equal = int(nx/size)

	nx_i_start = rank * nx_i_equal
	nx_i_end   = (rank + 1) * nx_i_equal

	if rank == size - 1 and nx_i_end != nx:
		nx_i_end += nx - size*nx_i_equal

	nx_i = nx_i_end - nx_i_start

	return nx_i_start, nx_i_end, nx_i

def distribute_reg_pairs(rank, n_reg, size):
	"""
 	get_reg_params_per_rank returns the index of the first and last regularization pair for each MPI rank
 
 	:rank: 		MPI rank 0, 1, ... ,p-1 
 	:n_reg: 	total number of regularization parameter pairs
 	:size: 		size of the MPI communicator (p in our case)

 	:return: the start and end indices, and the total number of snapshots for each MPI rank
 	"""

	nreg_i_equal = int(n_reg/size)

	start = rank * nreg_i_equal
	end   = (rank + 1) * nreg_i_equal

	if rank == size - 1 and end != n_reg:
		end += n_reg - size*nreg_i_equal

	return start, end

def compute_Qhat_sq(Qhat):
	"""
	compute_Qhat_sq returns the non-redundant terms in Qhat squared

	:Qhat: reduced data

	:return: Qhat_sq containing the non-redundant in Qhat squared
	"""

	if len(np.shape(Qhat)) == 1:

	    r 		= np.size(Qhat)
	    prods 	= []
	    for i in range(r):
	        temp = Qhat[i]*Qhat[i:]
	        prods.append(temp)

	    Qhat_sq = np.concatenate(tuple(prods))

	elif len(np.shape(Qhat)) == 2:
	    K, r 	= np.shape(Qhat)    
	    prods 	= []
	    
	    for i in range(r):
	        temp = np.transpose(np.broadcast_to(Qhat[:, i], (r - i, K)))*Qhat[:, i:]
	        prods.append(temp)
	    
	    Qhat_sq = np.concatenate(tuple(prods), axis=1)

	else:
	    print('invalid input!')

	return Qhat_sq

def compute_train_err(Qhat_train, Qtilde_train):
	"""
	compute_train_err computes the OpInf training error

	:Qhat_train: 	Qhat_trainerence data
	:Qtilde_train: 	Qtilde_train data

	:return: train_err containing the value of the training error
	"""
	train_err = np.max(np.sqrt(np.sum( (Qtilde_train - Qhat_train)**2, axis=1) / np.sum(Qhat_train**2, axis=1)))

	return train_err

def solve_opinf_difference_model(qhat0, n_steps_pred, dOpInf_red_model):
	"""
	solve_opinf_difference_model solves the discrete OpInf ROM for n_steps_pred over the target time horizon (training + prediction)

	:qhat0: 			reduced initial condition Qtilde0=np.matmul (Vr.T, q[:, 0]
	:n_steps_pred: 		number of steps over the target time horizon to solve the OpInf reduced model
	:dOpInf_red_model: 	dOpInf ROM

	:return: contains_nan flag indicating NaN presence in in the Qtilde_train reduced solution, Qtilde
	"""

	Qtilde    		= np.zeros((np.size(qhat0), n_steps_pred))
	contains_nans  	= False

	Qtilde[:, 0] = qhat0
	for i in range(n_steps_pred - 1):
	    Qtilde[:, i + 1] = dOpInf_red_model(Qtilde[:, i])

	if np.any(np.isnan(Qtilde)):
	    contains_nans = True

	return contains_nans, Qtilde.T

def compute_Qhat_cube(Qhat):

	if len(np.shape(Qhat)) == 1:

		state = Qhat

		state2 = compute_Qhat_sq(state)

		lens = special.binom(np.arange(2, len(state) + 2), 2).astype(int)
		
		Qhat_cube = np.concatenate(
			[state[i] * state2[: lens[i]] for i in range(state.shape[0])],
			axis=0,
			)

	elif len(np.shape(Qhat)) == 2:

		nt, r = Qhat.shape

		r_cube = r*(r + 1)*(r + 2) // 6

		Qhat_cube = np.zeros((nt, r_cube))

		for i in range(nt):
			state = Qhat[i, :]

			state2 = compute_Qhat_sq(state)

			lens = special.binom(np.arange(2, len(state) + 2), 2).astype(int)

			Qhat_cube[i, :] = np.concatenate(
				[state[i] * state2[: lens[i]] for i in range(state.shape[0])],
				axis=0,
				)

	return Qhat_cube

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
    r       = y.shape[0]
    yyt     = np.outer(y, y)  # (r x r)
    quad    = np.array([np.sum(Q[:, :, i] * yyt) for i in range(r)])

    # Linear + nonlinear combination
    dyrom = (nu * L3 + rho * L1) @ y + alp * quad
    
    return dyrom