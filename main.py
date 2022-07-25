import darts.models.physics.chemical
import numpy as np
import pandas as pd

import sys
sys.path.append('C:/Users/Jaro/Documents/inversemodelling/code_thesis/DARTS_1D_model/physics_sup_2/')

from model_Reaktoro import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from matplotlib import cm
import os
#import cProfile

redirect_darts_output('run.log')
n = Model()
n.init()
n.run_python(200, timestep_python=True)
n.print_timers()
n.print_stat()
# time_data = pd.DataFrame.from_dict(n.physics.engine.time_data)
# time_data.to_pickle("darts_time_data.pkl")
# n.save_restart_data()
# writer = pd.ExcelWriter('time_data.xlsx')
# time_data.to_excel(writer, 'Sheet1')
# writer.save()

# n.load_restart_data()
# time_data = pd.read_pickle("darts_time_data.pkl")

# if 0:
#     n.run_python(1, timestep_python=True)
#     n.print_timers()
#     n.print_stat()
#     time_data = pd.DataFrame.from_dict(n.physics.engine.time_data)
#     time_data.to_pickle("darts_time_data.pkl")
#     n.save_restart_data()
#     writer = pd.ExcelWriter('time_data.xlsx')
#     time_data.to_excel(writer, 'Sheet1')
#     writer.save()
#     n.print_and_plot('time_data.xlsx')
# else:
#     n.load_restart_data()
#     time_data = pd.read_pickle("darts_time_data.pkl")

""" plot results 2D """
Xn = np.array(n.physics.engine.X, copy=False)
nc = n.property_container.nc + n.thermal
ne = n.property_container.n_e
nb = n.reservoir.nb
P = Xn[0:ne*nb:ne]
# plt.plot(P)
# plt.show()
z1_darts = Xn[1:ne*nb:ne]
z2_darts = Xn[2:ne*nb:ne]
z3_darts = Xn[3:ne*nb:ne]
#z4_darts = Xn[4:ne*nb:ne]
# z5_darts = Xn[5:ne*nb:ne]
z4_darts = np.zeros(len(z1_darts))
for i in range(len(z1_darts)):
    z4_darts[i] = 1 - z1_darts[i] - z2_darts[i] - z3_darts[i]# - z4_darts[i]# - z5_darts[i]
nu, x, z_c, density = [], [], [], []
H2O, Ca, Na, Cl, OH, H, NaOH, Halite = [], [], [], [], [], [], [], []
for i in range(len(P)):
    z_e = [z1_darts[i], z2_darts[i], z3_darts[i], z4_darts[i]]#, z5_darts[i]]#, z6_darts[i]]
    nu_output, x_output, z_c_output, density_output = n.flash_properties(z_e, 320, P[i])  # itor
    OH.append(z_c_output[0])
    H.append(z_c_output[1])
    Na.append(z_c_output[2])
    Cl.append(z_c_output[3])
    H2O.append(z_c_output[4])
    #NaOH.append(z_c_output[5])
    #Cax.append(z_c_output[6])
    # Halite.append(z_c_output[7])
    nu.append(nu_output[0])
    x.append(x_output)
    z_c.append(z_c_output)
    density.append(density_output)

plt.figure(1)
plt.plot(P, label='pressure')
plt.legend()
plt.show()
plt.figure(2)
plt.plot(z1_darts, label='OH-')
plt.plot(z2_darts, label='H+')
plt.plot(z3_darts, label='Na+')
plt.plot(z4_darts, label='Cl-')
#plt.plot(z5_darts, label='X-')
# plt.yscale('log')
#plt.plot(z6_darts, label='CaX2')
plt.legend()
plt.ylabel('z_e')
plt.xlabel('x dimensionless')
plt.title('Composition', y=1)
plt.show()
plt.figure(3)
#plt.plot(H2O, label='H2O')
plt.plot(OH, label='OH-')
plt.plot(H, label='H+')
plt.plot(Na, label='Na+')
plt.plot(Cl, label='Cl-')
#plt.plot(NaOH, label='NaOH')
# plt.plot(X, label='X-')
# plt.plot(Nax, label='NaX')
# plt.plot(Cax, label='CaX2')
# plt.plot(Halite, label='Halite')
plt.legend()
plt.ylabel('z_c')
plt.xlabel('x dimensionless')
plt.title('Composition', y=1)
plt.show()
# plt.figure(4)
# plt.plot(nu)
# plt.ylabel('saturation')
# plt.xlabel('x dimensionless')
# plt.title('Water saturation', y=1)
# plt.show()