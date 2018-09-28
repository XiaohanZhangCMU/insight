import numpy as np
import matplotlib.pyplot as plt

pgtrain_data = np.loadtxt('pgtrain_loss_vs_iterations.txt')
no_pgtrain_data = np.loadtxt('no_pgtrain_loss_vs_iterations.txt')

plt.plot(pgtrain_data, 'o-', linewidth=4)
plt.plot(no_pgtrain_data, 'o-', linewidth=4)
plt.legend(('pgtrain (test acc=98.0%)', 'standard train (test acc=95.7%)'))
plt.title('Convergence rate: pgtrain vs standard')
plt.savefig('loss_comparsion.pdf');

plt.xlim(25,30)
plt.ylim(-0.0,0.1)
plt.savefig('loss_comparsion_zoomin.pdf');
