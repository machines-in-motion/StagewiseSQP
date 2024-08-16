'''
Plot the QP benchmarks
'''
import numpy as np

from plot_config import LABELS, COLORS, LINESTYLES
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


SAVE_PLOT = True

# Solvers
SOLVERS = ['CSQP',
           'OSQP',
           'HPIPM_DENSE', 
           'HPIPM_OCP']

PROBLEMS = ['solo12', 'Kuka', 'Taichi']


# if(name == 'Taichi'):
#     MAX_QP_TIME = int(1e5)     # in ms
#     MAX_QP_ITER = 1000
#     TIME_DISCRETIZATION = 1.  # the larger the faster (usefull for very fast problems) 
# if(name == 'Kuka'):
#     MAX_QP_TIME = int(1e1)       # in ms
#     MAX_QP_ITER = 100  
#     TIME_DISCRETIZATION = 0.01  # the larger the faster (usefull for very fast problems) 
# if(name == 'solo12'):
#     MAX_QP_TIME = int(5e3)     # in ms
#     MAX_QP_ITER = 100000
#     TIME_DISCRETIZATION = 1  # the larger the faster (usefull for very fast problems) 

npz_data = {}

# Load data and display timings
PREFIX = "data/"
for name in PROBLEMS:
    print("Problem : ", name)
    file_name      = PREFIX + name + "_qp_benchmark.npz"
    print("Loading " + file_name)
    npz_file = np.load(file_name)
    print("QP solving time ", name)
    print(" CSQP         : " , np.mean(npz_file['csqp_time_samples'])        ,  ' \xB1 ' , np.std(npz_file['csqp_time_samples']        )) #, ' | med = ', np.median(npz_file['csqp_time_samples']) ))
    print(" OSQP         : " , np.mean(npz_file['osqp_time_samples'])        ,  ' \xB1 ' , np.std(npz_file['osqp_time_samples']        )) #, ' | med = ', np.median(npz_file['osqp_time_samples']) ))
    print(" HPIPM_DENSE  : " , np.mean(npz_file['hpipm_dense_time_samples']) ,  ' \xB1 ' , np.std(npz_file['hpipm_dense_time_samples'] )) #, ' | med = ', np.median(npz_file['hpipm_dense_time_samples']) ))
    print(" HPIPM_OCP    : " , np.mean(npz_file['hpipm_ocp_time_samples'])   ,  ' \xB1 ' , np.std(npz_file['hpipm_ocp_time_samples']   )) #, ' | med = ', np.median(npz_file['hpipm_ocp_time_samples']) ))
    print("Number of QP iterations ", name)
    print(" CSQP         : " , np.mean(npz_file['csqp_iter_samples']) ,  ' \xB1 ' , np.std(npz_file['csqp_iter_samples']))
    print(" OSQP         : " , np.mean(npz_file['osqp_iter_samples']) ,  ' \xB1 ' , np.std(npz_file['osqp_iter_samples']))
    print(" HPIPM_DENSE  : " , np.mean(npz_file['hpipm_dense_iter_samples']) ,  ' \xB1 ' , np.std(npz_file['hpipm_dense_iter_samples']))
    print(" HPIPM_OCP    : " , np.mean(npz_file['hpipm_ocp_iter_samples']) ,  ' \xB1 ' , np.std(npz_file['hpipm_ocp_iter_samples']))
    print("Percentage of QP solved ", name)
    print(" CSQP         : " , 100*np.sum(npz_file['csqp_iter_solved_samples'])/(len(npz_file['csqp_iter_solved_samples'])))
    print(" OSQP         : " , 100*np.sum(npz_file['osqp_iter_solved_samples'])/(len(npz_file['csqp_iter_solved_samples'])))
    print(" HPIPM_DENSE  : " , 100*np.sum(npz_file['hpipm_dense_solved_samples'])/(len(npz_file['csqp_iter_solved_samples'])))
    print(" HPIPM_OCP    : " , 100*np.sum(npz_file['hpipm_ocp_solved_samples'])/(len(npz_file['csqp_iter_solved_samples'])))
    print("\n-----\n")

# if('solo12' in PROBLEMS):
#     quadrotor_name = PREFIX + "solo12.npz"
#     print("Loading " + quadrotor_name)
#     npz_quadrotor = np.load(quadrotor_name)
#     npz_data['Quadrotor'] = npz_quadrotor
#     N_SAMPLES_quadrotor = npz_quadrotor['N_SAMPLES']
#     MAXITER_quadrotor   = npz_quadrotor['MAXITER']
#     print("Average solving time per iteration Quadrotor \n")
#     print(" CSQP         = " , npz_quadrotor['ddp_mean_solve_time']         ,  ' \xB1 ' , npz_quadrotor['ddp_std_solve_time'])
#     print(" OSQP         = " , npz_quadrotor['fddp_mean_solve_time']        ,  ' \xB1 ' , npz_quadrotor['fddp_std_solve_time'])
#     print(" HPIPM_DENSE  = " , npz_quadrotor['fddp_filter_mean_solve_time'] ,  ' \xB1 ' , npz_quadrotor['fddp_filter_std_solve_time'])
#     print(" HPIPM_OCP    = " , npz_quadrotor['SQP_mean_solve_time']         ,  ' \xB1 ' , npz_quadrotor['SQP_std_solve_time'])

# if('Taichi' in PROBLEMS):
#     pendulum_name  = PREFIX + "Taichi.npz"
#     print("Loading " + pendulum_name)
#     npz_pendulum = np.load(pendulum_name)
#     npz_data['Pendulum'] = npz_pendulum
#     N_SAMPLES_pendulum = npz_pendulum['N_SAMPLES']
#     MAXITER_pendulum   = npz_pendulum['MAXITER']
#     print("Average solving time per iteration Pendulum \n")
#     print(" CSQP         = " , npz_pendulum['ddp_mean_solve_time']         ,  ' \xB1 ' , npz_pendulum['ddp_std_solve_time'])
#     print(" OSQP         = " , npz_pendulum['fddp_mean_solve_time']        ,  ' \xB1 ' , npz_pendulum['fddp_std_solve_time'])
#     print(" HPIPM_DENSE  = " , npz_pendulum['fddp_filter_mean_solve_time'] ,  ' \xB1 ' , npz_pendulum['fddp_filter_std_solve_time'])
#     print(" HPIPM_OCP    = " , npz_pendulum['SQP_mean_solve_time']         ,  ' \xB1 ' , npz_pendulum['SQP_std_solve_time'])


# solving_times = {
#     'DDP':         [1e3*npz_file['ddp_mean_solve_time'] , 1e3*npz_quadrotor['ddp_mean_solve_time'], 1e3*npz_pendulum['ddp_mean_solve_time'] , 1e3*npz_taichi['ddp_mean_solve_time'] ],
#     'FDDP':        [1e3*npz_file['fddp_mean_solve_time'] , 1e3*npz_quadrotor['fddp_mean_solve_time'], 1e3*npz_pendulum['fddp_mean_solve_time'] , 1e3*npz_taichi['fddp_mean_solve_time'] ],
#     'FDDP_filter': [1e3*npz_file['fddp_filter_mean_solve_time'] , 1e3*npz_quadrotor['fddp_filter_mean_solve_time'], 1e3*npz_pendulum['fddp_filter_mean_solve_time'] , 1e3*npz_taichi['fddp_filter_mean_solve_time'] ],
#     'SQP':         [1e3*npz_file['SQP_mean_solve_time'] , 1e3*npz_quadrotor['SQP_mean_solve_time'], 1e3*npz_pendulum['SQP_mean_solve_time'] , 1e3*npz_taichi['SQP_mean_solve_time'] ],
# }
