import numpy as np
import matplotlib.pyplot as plt

plot_axis = -1 # -1: loss 0: acc 1: val_acc


pgtrain_hist_24 = np.loadtxt('pg_train_results/pg_hist_24.dat')
pgtrain_hist_36 = np.loadtxt('pg_train_results/pg_hist_36.dat')
pgtrain_hist_48 = np.loadtxt('pg_train_results/pg_hist_48.dat')

pgtrain_time_24 = np.loadtxt('pg_train_results/pg_time_hist24.dat').reshape(-1,1)
pgtrain_time_36 = np.loadtxt('pg_train_results/pg_time_hist36.dat').reshape(-1,1)
pgtrain_time_48 = np.loadtxt('pg_train_results/pg_time_hist48.dat').reshape(-1,1)

no_pgtrain_hist = np.loadtxt('no_pg_train_results/no_pg_hist_48.dat')
no_pgtrain_time = np.loadtxt('no_pg_train_results/no_pg_time_hist48.dat').reshape(-1,1)


print(pgtrain_time_24.shape)
print(pgtrain_hist_24[:,-1].shape)
tmp1 = np.hstack((pgtrain_hist_24[:,plot_axis].reshape(-1,1),pgtrain_time_24))
tmp2 = np.hstack((pgtrain_hist_36[:,plot_axis].reshape(-1,1),pgtrain_time_36))
tmp3 = np.hstack((pgtrain_hist_48[:,plot_axis].reshape(-1,1),pgtrain_time_48))
pgtrain_loss_vs_time = np.vstack((tmp1, tmp2, tmp3))
print(pgtrain_loss_vs_time.shape)
tmp6 = pgtrain_loss_vs_time[:,1]
print(tmp6)
tmp7 = np.cumsum(tmp6)
print(tmp7)
pgtrain_loss_vs_time[:,1] = tmp7

no_pgtrain_loss_vs_time = np.vstack(np.hstack((no_pgtrain_hist[:,plot_axis].reshape(-1,1),no_pgtrain_time)))

tmp8 = no_pgtrain_loss_vs_time[:,1]
print(tmp8)
tmp9 = np.cumsum(tmp8)
print(tmp9)
no_pgtrain_loss_vs_time[:,1] = tmp9

plt.plot(pgtrain_loss_vs_time[:,1], pgtrain_loss_vs_time[:,0], 'o-', linewidth=2)
plt.plot(no_pgtrain_loss_vs_time[:,1], no_pgtrain_loss_vs_time[:,0], '*-', linewidth=2)
plt.xlabel('wall time clock (s)')
if plot_axis == -1:
    plt.ylabel('loss over validation set')
if plot_axis == 1:
    plt.ylabel('validation accuracy')

plt.legend(('pgtrain (test acc=98.0%)', 'standard train (test acc=95.7%)'))
plt.title('Convergence rate: pgtrain vs standard')
plt.savefig('loss_comparsion.png');
#plt.xlim(600,1000)
plt.ylim(-0.0,0.1)
plt.savefig('loss_comparsion_zoomin.png');
