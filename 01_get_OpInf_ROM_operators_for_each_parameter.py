from utils.utils import *
from config.config import *

import scipy.io as sio


if __name__ == '__main__':

	mat_data = sio.loadmat(trajectory_file_500)
	Q_train = mat_data["y"].T
	# Q_train = Q_train[:, :nt]

	print(Q_train.shape)

	######## INITIALIZATION ########
	# compute the Cartesian product of all regularization pairs (beta1, beta2 )
	reg_pairs_global 	= list(product(B1, B2))
	n_reg_global 		= len(reg_pairs_global)

	print(n_reg_global)
	###### INITIALIZATION END ######

	start_time_global = time()

	######## STEP I: SEQUENTIAL DATA TRANSFORMATIONS ########
	compute_time 					= 0	
	communication_time 				= 0
	start_time_data_transformations = time()

	# if CENTERING:
	# 	# compute the global temporal mean of each variable
	# 	temporal_mean_global 	= np.mean(Q_train, axis=1)
	# 	# center (in place) each variable with respect to its global temporal mean
	# 	Q_train 				-= temporal_mean_global[:, np.newaxis]

	end_time_data_transformations 	= time()
	compute_time					+= end_time_data_transformations - start_time_data_transformations
	#################### STEP I END ##########################

	
	######## STEP II: SEQUENTIAL DIMENSIONALITY REDUCTION ########
	start_time_matmul 	= time()
	# compute the local Gram matrices on each rank
	D_global  			= np.matmul(Q_train.T, Q_train)
	end_time_matmul 	= time()
	compute_time		+= end_time_matmul - start_time_matmul
	
	start_time = time()
	
	# compute the eigendecomposition of the positive, semi-definite global Gram matrix
	eigs, eigv = np.linalg.eigh(D_global)

	# order eigenpairs by increasing eigenvalue magnitude
	sorted_indices 	= np.argsort(eigs)[::-1]
	eigs 			= eigs[sorted_indices]
	eigv 			= eigv[:, sorted_indices]

	# compute retained energy for r bteween 1 and nt
	ret_energy 	= np.cumsum(eigs)/np.sum(eigs)
	# select reduced dimension r for that the retained energy exceeds the prescribed threshold
	r 			= np.argmax(ret_energy > target_energy)

	print('r = ', r)

	# compute the auxiliary Tr matrix
	Tr_global 	= np.matmul(eigv[:, :r], np.diag(eigs[:r]**(-0.5)))
	# compute the low-dimensional representation of the high-dimensional transformed snapshot data
	Qhat_global = np.matmul(Tr_global.T, D_global)

	end_time 		= time()
	compute_time	+= end_time - start_time
	##################### STEP II END #############################


	######## STEP III: SEQUENTIAL REDUCED OPERATOR INFERENCE ########
	learning_time_grid_search_total = 0	
	start_time_grid_search_total 	= time()

	# extract left and right shifted reduced data matrices for the discrete OpInf learning problem
	Qhat_1 = Qhat_global.T[:-1, :]
	Qhat_2 = Qhat_global.T[1:, :]

	# column dimension of the data matrix Dhat used in the discrete OpInf learning problem
	s = int(r*(r + 1)/2)
	d = 2*r + s

	# compute the non-redundant quadratic terms of Qhat_1 squared
	Qhat_1_sq = compute_Qhat_sq(Qhat_1)

	# assemble the data matrix Dhat for the discrete OpInf learning problem
	Dhat   = np.concatenate((nu*Qhat_1, rho*Qhat_1, alp*Qhat_1_sq), axis=1)
	# compute Dhat.T @ Dhat for the normal equations to solve the OpInf least squares minimization
	Dhat_2 = Dhat.T @ Dhat

	# compute the temporal mean and maximum deviation of the reduced training data
	mean_Qhat_train   	 	= np.mean(Qhat_global.T, axis=0)
	max_diff_Qhat_train 	= np.max(np.abs(Qhat_global.T - mean_Qhat_train), axis=0)
	# training error corresponding to the optimal regularization hyperparameters
	opt_train_err 			= 1e20


	Phir_global = np.matmul(Q_train, Tr_global)

	# loop over the regularization pairs corresponding to each MPI rank
	for pair in reg_pairs_global:

		# extract beta1 and beta2 from each candidate regularization pair
		beta1 = pair[0]
		beta2 = pair[1]

		start_time_OpInf_learning = time()

		# regularize the linear and constant reduced operators using beta1, and the quadratic operator using beta2
		regg            	= np.zeros(d)
		regg[:2*r]        	= beta1
		regg[2*r : 2*r + s] = beta2
		regularizer     	= np.diag(regg)
		Dhat_2_reg 			= Dhat_2 + regularizer

		# solve the OpInf learning problem by solving the regularized normal equations
		Ohat = np.linalg.solve(Dhat_2_reg, np.dot(Dhat.T, Qhat_2)).T

		# extract the linear, quadratic, and constant reduced model operators
		Ahat1 	= Ohat[:, :r]
		Ahat2 	= Ohat[:, r:2*r]
		Fhat 	= Ohat[:, 2*r:2*r + s]

		# print(Ahat1)
		
		end_time_OpInf_learning = time()

		# define the OpInf reduced model 
		dOpInf_red_model 	= lambda x: nu*Ahat1 @ x + rho*Ahat2 @ x + alp*Fhat @ compute_Qhat_sq(x)
		# extract the reduced initial condition from Qhat_1
		qhat0 				= Qhat_1[0, :]
		
		# compute the reduced solution over the trial time horizon, which here is the same as the target time horizon
		start_time_OpInf_eval 			= time()
		contains_nans, Qtilde_OpInf 	= solve_opinf_difference_model(qhat0, nt_all, dOpInf_red_model)
		end_time_OpInf_eval 			= time()

		time_OpInf_learning = end_time_OpInf_learning - start_time_OpInf_learning
		time_OpInf_eval 	= end_time_OpInf_eval - start_time_OpInf_eval

		learning_time_grid_search_total += time_OpInf_learning
		
		# for each candidate regulairzation pair, we compute the training error 
		# we also save the corresponding reduced solution, learning time and ROM evaluation time
 		# and compute the ratio of maximum coefficient growth in the trial period to that in the training period
		if contains_nans == False:
			train_err     			= compute_train_err(Qhat_global.T[:nt, :], Qtilde_OpInf[:nt, :])
			max_diff_Qhat_trial  	= np.max(np.abs(Qtilde_OpInf - mean_Qhat_train), axis=0)			
			max_growth_trial  		= np.max(max_diff_Qhat_trial)/np.max(max_diff_Qhat_train)

			print(pair, train_err, max_growth_trial)

			Q_OpInf = Phir_global @ Qtilde_OpInf.T

			print(np.linalg.norm(Q_OpInf[:, :nt] - Q_train[:, :nt])/np.linalg.norm(Q_train[:, :nt]))
			print(np.linalg.norm(Q_OpInf[:, nt:] - Q_train[:, nt:])/np.linalg.norm(Q_train[:, nt:]))

			if max_growth_trial < max_growth:

				if train_err < opt_train_err:
					opt_train_err 				= train_err
					Qtilde_OpInf_opt 			= Qtilde_OpInf
					OpInf_wtime_learning_opt	= time_OpInf_learning
					OpInf_ROM_wtime_opt			= time_OpInf_eval

					beta1_opt = pair[0]
					beta2_opt = pair[1]

					Ahat1_opt 	= Ahat1.copy()
					Ahat2_opt 	= Ahat2.copy()
					Fhat_opt 	= Fhat.copy()

	end_time_grid_search_total 		= time()
	compute_time_grid_search_total	= end_time_grid_search_total - start_time_grid_search_total
	####################### STEP III END #############################

	print(beta1_opt, beta2_opt, opt_train_err)
	
	Q_OpInf = Phir_global @ Qtilde_OpInf_opt.T

	print(np.linalg.norm(Q_OpInf[:, :nt] - Q_train[:, :nt])/np.linalg.norm(Q_train[:, :nt]))

	# np.save(OpInf_ROM_sol_file, Q_OpInf)

	print(Ahat1_opt)
	print(Ahat2_opt)
	print(Fhat_opt)

	np.savez('data/OpInf_ROM_operators.npz', Ahat1=Ahat1_opt, Ahat2=Ahat2_opt, Fhat=Fhat_opt)
	np.save('data/OpInf_POD_basis.npy', Phir_global)