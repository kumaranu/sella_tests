import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.integrate import LSODA


def dvdt(t, v):
    print('called')
    return 3*v**2 - 5


v0 = 0
t = np.linspace(0, 1, 100)
#sol_m1 = odeint(dvdt, y0=v0, t=t, tfirst=True)
#sol_m2 = solve_ivp(dvdt, t_span=(0, max(t)), y0=[v0], t_eval=t)
print('Before')

sol_m3 = LSODA(dvdt, t0=0, y0=[v0], t_bound=1)
print('After')

#v_sol_m1 = sol_m1.T[0]
#v_sol_m2 = sol_m2.y[0]

v_sol_m3 = []
while sol_m3.status == 'running':
    sol_m3.step()
    v_sol_m3.append([sol_m3.t, sol_m3.y[0]])
    print(f'nfev: {sol_m3.nfev}')
# v_sol_m3 = sol_m3.y[0]

#print(len(v_sol_m1))
#print(len(v_sol_m2))
print(len(v_sol_m3))

v_sol_m3 = np.asarray(v_sol_m3)

#plt.plot(t, v_sol_m1)
#plt.plot(t, v_sol_m2, '--')
plt.plot(v_sol_m3[:, 0], v_sol_m3[:, 1], '*')
plt.ylabel('$v(t)$', fontsize=22)
plt.xlabel('$t$', fontsize=22)
plt.show()

