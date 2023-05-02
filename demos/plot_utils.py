import numpy as np
import matplotlib.pyplot as plt 


# Plot
class SimpleDataPlotter:
    def __init__(self, dt=1e-3):
        self.dt = dt

    def plot_trajs(self, trajs, labels, colors, ylims=None, markers=None, linestyle=None):
      '''
      Plot trajectories
      '''   
      # Check input trajs
      if type(trajs != list):
        print('trajs must be list')
      N = trajs[0].shape[0] - 1
      n = trajs[0].shape[1]
      for k,tr in enumerate(trajs):
        if(trajs[k].shape[0] - 1 != N):
          print('error: traj '+str(k)+' has wrong size N='+str(N))
        if(trajs[k].shape[1] != n):
          print('error: traj '+str(k)+' has wrong size n='+str(n))
      tspan = np.linspace(0, N*self.dt, N+1)      
      fig, ax = plt.subplots(n, 1, sharex='col')  
      for i in range(n):
          for k,tr in enumerate(trajs):
            if(markers is not None and markers[k] is not None): mark = markers[k]
            else: mark = None
            if(linestyle is not None and linestyle[k] is not None): line = linestyle[k]
            else: line = '-'
            ax[i].plot(tspan, tr[:,i], linestyle=line, marker=mark, label=labels[k], color=colors[k], alpha=0.9)
          ax[i].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
          ax[i].grid(True)
          if(markers is not None and markers[k] is not None): mark = markers[k]
          else: mark = None
          if(ylims is not None):
            ax[i].set_ylim(ylims[0][i], ylims[1][i])
      ax[-1].set_xlabel('Time (s)', fontsize=16)
      fig.align_ylabels(ax[:])
      handles, labels = ax[0].get_legend_handles_labels()
      fig.legend(handles, labels, loc='upper right', prop={'size': 16})
      fig.align_ylabels()
      return fig, ax

    def plot_joint_pos(self, trajs, labels, colors, ylims=None, markers=None, linestyle=None):
      fig, ax = self.plot_trajs(trajs, labels, colors, ylims=ylims, markers=markers, linestyle=linestyle)
      for i in range(len(ax)):
        ax[i].set_ylabel('$q_%s$'%i, fontsize=16)
      fig.suptitle('Joint position trajectories', size=18)
      return fig, ax

    def plot_joint_vel(self, trajs, labels, colors, ylims=None, markers=None, linestyle=None):
      fig, ax = self.plot_trajs(trajs, labels, colors, ylims=ylims, markers=markers, linestyle=linestyle)
      for i in range(len(ax)):
        ax[i].set_ylabel('$v_%s$'%i, fontsize=16)
      fig.suptitle('Joint velocity trajectories', size=18)
      return fig, ax

    def plot_joint_tau(self, trajs, labels, colors, ylims=None, markers=None, linestyle=None, title=None):
      fig, ax = self.plot_trajs(trajs, labels, colors, ylims=ylims, markers=markers, linestyle=linestyle)
      for i in range(len(ax)):
        ax[i].set_ylabel('$tau_%s$'%i, fontsize=16)
        if(title == None):
          fig.suptitle('Joint torque trajectories', size=18)
        else:
          fig.suptitle(title, size=18)
      return fig, ax

    def plot_ee_pos(self, trajs, labels, colors, ylims=None, markers=None, linestyle=None):
      fig, ax = self.plot_trajs(trajs, labels, colors, ylims=ylims, markers=markers, linestyle=linestyle)
      for i in range(len(ax)):
        ax[i].set_ylabel('$p_%s$'%i, fontsize=16)
      fig.suptitle('End-effector position', size=18)
      return fig, ax

    def plot_ee_vel(self, trajs, labels, colors, ylims=None, markers=None, linestyle=None):
      fig, ax = self.plot_trajs(trajs, labels, colors, ylims=ylims, markers=markers, linestyle=linestyle)
      for i in range(len(ax)):
        ax[i].set_ylabel('$pdot_%s$'%i, fontsize=16)
      fig.suptitle('End-effector linear velocity', size=18)
      return fig, ax

    def plot_ee_rpy(self, trajs, labels, colors, ylims=None, markers=None, linestyle=None):
      fig, ax = self.plot_trajs(trajs, labels, colors, ylims=ylims, markers=markers, linestyle=linestyle)
      names = ['Roll', 'Pitch', 'Yaw']
      for i in range(len(ax)):
        ax[i].set_ylabel(names[i], fontsize=16)
      fig.suptitle('End-effector orientation (RPY)', size=18)
      return fig, ax

    def plot_ee_w(self, trajs, labels, colors, ylims=None, markers=None, linestyle=None):
      fig, ax = self.plot_trajs(trajs, labels, colors, ylims=ylims, markers=markers, linestyle=linestyle)
      for i in range(len(ax)):
        ax[i].set_ylabel('$\\omega_%s$'%i, fontsize=16)
      fig.suptitle('End-effector angular velocity', size=18)
      return fig, ax

    def plot_mpc_solve_time(self, timings):
      # Parameters
      N = ts.shape[0]-1
      nq = ts.shape[1]
      # Plots
      tspan = np.linspace(0, N*self.dt, N+1)
      fig, ax = plt.subplots(nq, 1, sharex='col')  
      ax[i].plot(tspan, ts[:,i], linestyle='-', marker='.', label='Actual', color=[1., 0., 0.], alpha=0.9)
      ax[i].set_ylabel('$v_%s$'%i, fontsize=16)
      ax[i].yaxis.set_major_locator(plt.MaxNLocator(2))
      ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
      ax[i].grid(True)
      ax[-1].set_xlabel('Time (s)', fontsize=16)
      fig.align_ylabels(ax[:])
      handles, labels = ax[0].get_legend_handles_labels()
      fig.legend(handles, labels, loc='upper right', prop={'size': 16})
      fig.align_ylabels()
      fig.suptitle('Joint torque trajectories', size=18)
      return fig, ax

    def plot_soft_contact_force(self, trajs, labels, colors, ylims=None, markers=None, linestyle=None):
      '''
      Plot soft contact force (3d)
      '''   
      fig, ax = self.plot_trajs(trajs, labels, colors, ylims=ylims, markers=markers, linestyle=linestyle)
      for i in range(len(ax)):
        ax[i].set_ylabel('$\\lambda_%s$'%i, fontsize=16)
      fig.suptitle('Contact force', size=18)
      return fig, ax
