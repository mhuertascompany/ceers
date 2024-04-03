from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt


a = beta(1,5)
b = beta(2,6)


x = np.linspace(0,1,200)
pa = a.pdf(x)
pb = b.pdf(x)
pc = (pa + pb) / 2

plt.xlim((0,1))
plt.ylim((0,5))
plt.plot(x,pa)
plt.plot(x,pb)
plt.plot(x,pc)
plt.savefig('test_dirichlet.png')