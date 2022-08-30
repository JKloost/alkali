import darts.models.physics.chemical
import numpy as np
import pandas as pd

import sys
sys.path.append('C:/Users/Jaro/Documents/inversemodelling/code_thesis/DARTS_1D_model/physics_sup_2/')

from model_Reaktoro import Model
# from model_phreeqc import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from matplotlib import cm
import os
# import cProfile

redirect_darts_output('run.log')
n = Model()
n.init()
n.run_python(20000, timestep_python=False)
n.print_timers()
n.print_stat()
time_data = pd.DataFrame.from_dict(n.physics.engine.time_data)
time_data.to_pickle("darts_time_data.pkl")
n.save_restart_data()
writer = pd.ExcelWriter('time_data.xlsx')
time_data.to_excel(writer, 'Sheet1')
writer.save()

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
# print('Xn', Xn)
z_darts = np.zeros((ne, len(P)))
for i in range(1, ne):
    z_darts[i-1] = Xn[i:ne*nb:ne]
z_darts[-1] = np.ones(len(P)) - list(map(sum, zip(*z_darts[:-1])))
# print('z_darts', z_darts)
nu, x, z_c, density, pH = [], [], [], [], []
H2O, Ca, Na, Cl, OH, H, NaCO3, CO3, HCO3, NaSO4, CaSO4, SO4, K = [], [], [], [], [], [], [], [], [], [], [], [], []
for i in range(len(P)):
    # print('input', z_darts[:,i])
    nu_output, x_output, z_c_output, density_output, pH_output = n.flash_properties(z_darts[:, i], 273+20, P[i])  # itor
    # print('pH', pH_output)
    H2O.append(z_c_output[0])
    Na.append(z_c_output[1])
    Ca.append(z_c_output[2])
    SO4.append(z_c_output[3])
    Cl.append(z_c_output[4])
    # K.append(z_c_output[5])
    NaSO4.append(z_c_output[5])
    # H2CO3.append(z_c_output[7])
    # NaCO3.append(z_c_output[8])
    # NaOH.append(z_c_output[9])
    # NaHCO3.append(z_c_output[10])
    CaSO4.append(z_c_output[-1])
    #NaOH.append(z_c_output[5])
    #Cax.append(z_c_output[6])
    #Halite.append(z_c_output[5])
    nu.append(nu_output[0])
    x.append(x_output)
    z_c.append(z_c_output)
    density.append(density_output)
    pH.append(pH_output)

plt.figure(1)
plt.plot(P, label='pressure')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(z_darts[0], label='H2O')
plt.plot(z_darts[1], label='Na+')
plt.plot(z_darts[2], label='Ca+2')
plt.plot(z_darts[3], label='Cl-')
plt.plot(z_darts[4], label='SO4-2')
plt.legend(loc=4)
plt.ylabel('z_e')
plt.yscale('log')
plt.xlabel('x dimensionless')
plt.title('Composition', y=1)
plt.show()

plt.figure(3)
plt.plot(H2O, label='H2O')
plt.plot(Na, label='Na+')
plt.plot(Ca, label='Ca+2')
plt.plot(SO4, label='SO4-2')
plt.plot(Cl, label='Cl-')
plt.plot(NaSO4, label='NaSO4-')
plt.plot(CaSO4, label='CaSO4')
# plt.plot(H2CO3, label='H2CO3')
# plt.plot(NaCO3, label='NaCO3')
# plt.plot(NaHCO3, label='NaHCO3-')
# plt.plot(NaOH, label='NaOH')
# plt.plot(K, label='K+')
plt.plot()
# plt.plot(Halite, label='Halite')
plt.yscale('log')
plt.legend(loc=4)
# plt.legend()
plt.ylabel('z_c')
plt.xlabel('x dimensionless')
plt.title('Composition', y=1)
plt.show()

# plt.figure(4)
# plt.plot(pH)
# plt.ylabel('pH')
# plt.xlabel('x dimensionless')
# plt.title('pH', y=1)
# plt.show()
