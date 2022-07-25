from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import sim_params
import numpy as np
from properties_basic import *
from property_container import *
from reaktoro import *  # reaktoro v2.0.0rc22
from physics_comp_sup import Compositional
import matplotlib.pyplot as plt


class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.reaktoro = Reaktoro()  # Initialise Reaktoro
        self.db = PhreeqcDatabase('phreeqc.dat')
        self.zero = 1e-11
        perm = 100  # / (1 - solid_init) ** trans_exp
        nx = 100
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=1, dy=10, dz=1, permx=perm, permy=perm,
                                         permz=perm/10, poro=1, depth=1000)

        # """well location"""
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=1, multi_segment=False)

        self.reservoir.add_well("P1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=nx, j=1, k=1, multi_segment=False)

        """Physical properties"""
        # Create property containers:
        components_name = ['OH-', 'H+', 'Na+', 'Cl-', 'H2O']
        elements_name = ['OH-', 'H+', 'Na+', 'Cl-']
        # aqueous_phase = ['H2O(aq)', 'CO2(aq)', 'Ca+2', 'CO3-2', 'Na+', 'Cl-']
        # gas_phase = ['H2O(g)', 'CO2(g)']
        # solid_phase = ['Calcite', 'Halite']

        # E_mat = np.array([[1, 0, 0, 0, 0, 0, 0],
        #                   [0, 1, 0, 0, 0, 0, 1],
        #                   [0, 0, 1, 0, 0, 1, 0],
        #                   [0, 0, 0, 1, 0, 0, 0],
        #                   [0, 0, 0, 0, 1, 1, 2]])
        # E_mat = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #                   [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        #                   [0, 0, 0, 0, 1, 0, 0, 1, 0, 4, 0, 0, 0],
        #                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        #                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        #                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        E_mat = np.array([[1, 0, 0, 0, 1],
                          [0, 1, 0, 0, 1],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0]])
        # E_mat = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #                     [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        #                   [0, 0, 1, 0, 0, 1, 0, 4, 0, 0, 0],
        #                   [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        #                   [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        #                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        Mw = np.zeros(len(components_name))
        for i in range(len(Mw)):
            component = Species(str(components_name[i]))
            Mw[i] = component.molarMass()*1000
        aq = AqueousPhase(StringList(['OH-', 'H+', 'Na+', 'Cl-', 'H2O']))
        system_inj, system_ini = ChemicalSystem(self.db, aq), ChemicalSystem(self.db, aq)
        state_inj, state_ini = ChemicalState(system_inj), ChemicalState(system_ini)
        specs_inj, specs_ini = EquilibriumSpecs(system_inj), EquilibriumSpecs(system_ini)
        specs_inj.temperature(), specs_ini.temperature()
        specs_inj.pressure(), specs_ini.pressure()
        specs_inj.pH(), specs_ini.pH()
        solver_inj = EquilibriumSolver(specs_inj)
        solver_ini = EquilibriumSolver(specs_ini)

        conditions_inj, conditions_ini = EquilibriumConditions(specs_inj), EquilibriumConditions(specs_ini)

        conditions_inj.temperature(350, 'kelvin'),  conditions_ini.temperature(350, 'kelvin')
        conditions_inj.pressure(200, 'bar'),        conditions_ini.pressure(200, 'bar')
        conditions_inj.pH(1),                       conditions_ini.pH(7)

        state_inj.set('H2O', 1, 'kg'),              state_ini.set('H2O', 1, 'kg')
        state_inj.set('Na+', 4.5, 'mol'),           state_ini.set('Na+', 1, 'mol')
        state_inj.set('Cl-', 4.5, 'mol'),           state_ini.set('Cl-', 1, 'mol')

        result_inj = solver_inj.solve(state_inj, conditions_inj)
        result_ini = solver_ini.solve(state_ini, conditions_ini)
        cp_inj = ChemicalProps(state_inj)
        z_c_inj = [float(cp_inj.speciesAmount(0)), float(cp_inj.speciesAmount(1)), float(cp_inj.speciesAmount(2)),
                   float(cp_inj.speciesAmount(3)), float(cp_inj.speciesAmount(4))]
        for i in range(len(z_c_inj)):
            if z_c_inj[i] < self.zero:
                z_c_inj[i] = 0
        z_c_inj = [float(i) / sum(z_c_inj) for i in z_c_inj]
        z_e_inj = np.zeros(E_mat.shape[0])
        for i in range(E_mat.shape[0]):
            z_e_inj[i] = np.divide(np.sum(np.multiply(E_mat[i], z_c_inj)), np.sum(np.multiply(E_mat, z_c_inj)))

        cp_ini = ChemicalProps(state_ini)
        z_c_ini = [float(cp_ini.speciesAmount(0)), float(cp_ini.speciesAmount(1)), float(cp_ini.speciesAmount(2)),
                   float(cp_ini.speciesAmount(3)), float(cp_ini.speciesAmount(4))]
        for i in range(len(z_c_ini)):
            if z_c_ini[i] < self.zero:
                z_c_ini[i] = 0
        z_c_ini = [float(i) / sum(z_c_ini) for i in z_c_ini]
        z_e_ini = np.zeros(E_mat.shape[0])
        for i in range(E_mat.shape[0]):
            z_e_ini[i] = np.divide(np.sum(np.multiply(E_mat[i], z_c_ini)), np.sum(np.multiply(E_mat, z_c_ini)))

        self.thermal = 0
        # solid_density = [2000, 2000]  # fill in density for amount of solids present
        solid_density = None
        self.property_container = model_properties(phases_name=['wat'],
                                                   components_name=components_name, elements_name=elements_name,
                                                   reaktoro=self.reaktoro, E_mat=E_mat, diff_coef=1e-9, rock_comp=1e-7,
                                                   Mw=Mw, min_z=self.zero / 10, solid_dens=solid_density)

        """ properties correlations """
        # self.property_container.flash_ev = Flash(self.components[:-1], [10, 1e-12, 1e-1], self.zero)
        # return [y, x], [V, 1-V]
        self.property_container.density_ev = dict([
            ('wat', Density(compr=1e-6, dens0=1000))])

        self.property_container.viscosity_ev = dict([
            ('wat', ViscosityConst(1))])
        self.property_container.rel_perm_ev = dict([
            ('wat', PhaseRelPerm("wat"))])

        # ne = self.property_container.nc + self.thermal
        # self.property_container.kinetic_rate_ev = kinetic_basic(equi_prod, 1e-0, ne)

        """ Activate physics """
        self.physics = Compositional(self.property_container, self.timer, n_points=101, min_p=1, max_p=1000,
                                     min_z=self.zero / 10, max_z=1 - self.zero / 10, cache=0)

        #                  H2O,                     CO2,    Ca++,       CO3--,      Na+, Cl-
        #                  Oh- H+ Na+ Cl-
        self.ini_stream = z_e_ini[:-1]
        self.inj_stream = z_e_inj[:-1]
        # self.ini_stream = [0.5-self.zero, 0.5, self.zero]  # 0.002
        # self.inj_stream = [0.51-self.zero, 0.49, self.zero]  # 0.000
        # self.inj_stream = self.ini_stream

        self.params.first_ts = 1e-6
        self.params.max_ts = 10
        self.params.mult_ts = 2

        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-6
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop
        # self.params.newton_params[0] = 0.2

        self.timer.node["initialization"].stop()

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        """ initialize conditions for all scenarios"""
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, 200, self.ini_stream)
        # volume = np.array(self.reservoir.volume, copy=False)
        # volume.fill(100)
        # volume[0] = 1e10
        # volume[-1] = 1e10
        # pressure = np.array(self.reservoir.mesh.pressure, copy=False)
        # pressure.fill(200)
        # pressure[0] = 230
        # pressure[-1] = 170
        # comp = np.array(self.reservoir.mesh.composition, copy=False)
        # comp[0] = self.inj_stream[0]
        # comp[1] = self.inj_stream[1]
        # comp[2] = self.inj_stream[2]
        # comp[3] = self.inj_stream[3]
        # comp[4] = self.inj_stream[4]
        # composition = np.array(self.reservoir.mesh.composition, copy=False)
        # n_half = int(self.reservoir.nx * self.reservoir.ny * self.reservoir.nz / 2)
        # composition[2*n_half:] = 1e-6

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                # w.control = self.physics.new_rate_inj(0.2, self.inj_stream, 0)
                w.control = self.physics.new_bhp_inj(210, self.inj_stream)
            else:
                w.control = self.physics.new_bhp_prod(190)

    def set_op_list(self):
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks
        self.op_num[n_res:] = 1
        self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_w_itor]

    def properties(self, state):
        (sat, x, rho, rho_m, mu, kr, ph) = self.property_container.evaluate(state)
        return sat[0]

    def flash_properties(self, ze, T, P):
        nu, x, zc, density = Flash_Reaktoro(ze, T, P, self.reaktoro)
        return nu, x, zc, density

    def print_and_plot(self, filename):
        nc = self.property_container.nc
        Sg = np.zeros(self.reservoir.nb)
        Ss = np.zeros(self.reservoir.nb)
        X = np.zeros((self.reservoir.nb, nc - 1, 2))

        rel_perm = np.zeros((self.reservoir.nb, 2))
        visc = np.zeros((self.reservoir.nb, 2))
        density = np.zeros((self.reservoir.nb, 3))
        density_m = np.zeros((self.reservoir.nb, 3))

        Xn = np.array(self.physics.engine.X, copy=True)

        P = Xn[0:self.reservoir.nb * nc:nc]
        z_caco3 = 1 - (
                    Xn[1:self.reservoir.nb * nc:nc] + Xn[2:self.reservoir.nb * nc:nc] + Xn[3:self.reservoir.nb * nc:nc])

        z_co2 = Xn[1:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_inert = Xn[2:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_h2o = Xn[3:self.reservoir.nb * nc:nc] / (1 - z_caco3)

        for ii in range(self.reservoir.nb):
            x_list = Xn[ii * nc:(ii + 1) * nc]
            state = value_vector(x_list)
            (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property_container.evaluate(state)

            rel_perm[ii, :] = kr
            visc[ii, :] = mu
            density[ii, :2] = rho
            density_m[ii, :2] = rho_m

            density[2] = self.property_container.solid_dens[-1]

            X[ii, :, 0] = x[1][:-1]
            X[ii, :, 1] = x[0][:-1]
            Sg[ii] = sat[0]
            Ss[ii] = z_caco3[ii]

        # Write all output to a file:
        with open(filename, 'w+') as f:
            # Print headers:
            print(
                '//Gridblock\t gas_sat\t pressure\t C_m\t poro\t co2_liq\t co2_vap\t h2o_liq\t h2o_vap\t '
                'ca_plus_co3_liq\t liq_dens\t vap_dens\t solid_dens\t liq_mole_dens\t vap_mole_dens\t solid_mole_dens'
                '\t rel_perm_liq\t rel_perm_gas\t visc_liq\t visc_gas',
                file=f)
            print(
                '//[-]\t [-]\t [bar]\t [kmole/m3]\t [-]\t [-]\t [-]\t [-]\t [-]\t [-]\t [kg/m3]\t [kg/m3]\t [kg/m3]\t '
                '[kmole/m3]\t [kmole/m3]\t [kmole/m3]\t [-]\t [-]\t [cP]\t [cP]',
                file=f)
            for ii in range(self.reservoir.nb):
                print(
                    '{:d}\t {:6.5f}\t {:7.5f}\t {:7.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t '
                    '{:8.5f}\t {:8.5f}\t {:8.5f}\t {:7.5f}\t {:7.5f}\t {:7.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t '
                    '{:6.5f}'.format(
                        ii, Sg[ii], P[ii], Ss[ii] * density_m[ii, 2], 1 - Ss[ii], X[ii, 0, 0], X[ii, 0, 1], X[ii, 2, 0],
                        X[ii, 2, 1], X[ii, 1, 0],
                        density[ii, 0], density[ii, 1], density[ii, 2], density_m[ii, 0], density_m[ii, 1],
                        density_m[ii, 2],
                        rel_perm[ii, 0], rel_perm[ii, 1], visc[ii, 0], visc[ii, 1]), file=f)

        """ start plots """

        font_dict_title = {'family': 'sans-serif',
                           'color': 'black',
                           'weight': 'normal',
                           'size': 14,
                           }

        font_dict_axes = {'family': 'monospace',
                          'color': 'black',
                          'weight': 'normal',
                          'size': 14,
                          }

        fig, axs = plt.subplots(3, 3, figsize=(12, 10), dpi=200, facecolor='w', edgecolor='k')
        """ sg and x """
        axs[0][0].plot(z_co2, 'b')
        axs[0][0].set_xlabel('x [m]', font_dict_axes)
        axs[0][0].set_ylabel('$z_{CO_2}$ [-]', font_dict_axes)
        axs[0][0].set_title('Fluid composition', fontdict=font_dict_title)

        axs[0][1].plot(z_h2o, 'b')
        axs[0][1].set_xlabel('x [m]', font_dict_axes)
        axs[0][1].set_ylabel('$z_{H_2O}$ [-]', font_dict_axes)
        axs[0][1].set_title('Fluid composition', fontdict=font_dict_title)

        axs[0][2].plot(z_inert, 'b')
        axs[0][2].set_xlabel('x [m]', font_dict_axes)
        axs[0][2].set_ylabel('$z_{w, Ca+2} + z_{w, CO_3-2}$ [-]', font_dict_axes)
        axs[0][2].set_title('Fluid composition', fontdict=font_dict_title)

        axs[1][0].plot(X[:, 0, 0], 'b')
        axs[1][0].set_xlabel('x [m]', font_dict_axes)
        axs[1][0].set_ylabel('$x_{w, CO_2}$ [-]', font_dict_axes)
        axs[1][0].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[1][1].plot(X[:, 2, 0], 'b')
        axs[1][1].set_xlabel('x [m]', font_dict_axes)
        axs[1][1].set_ylabel('$x_{w, H_2O}$ [-]', font_dict_axes)
        axs[1][1].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[1][2].plot(X[:, 1, 0], 'b')
        axs[1][2].set_xlabel('x [m]', font_dict_axes)
        axs[1][2].set_ylabel('$x_{w, Ca+2} + x_{w, CO_3-2}$ [-]', font_dict_axes)
        axs[1][2].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[2][0].plot(P, 'b')
        axs[2][0].set_xlabel('x [m]', font_dict_axes)
        axs[2][0].set_ylabel('$p$ [bar]', font_dict_axes)
        axs[2][0].set_title('Pressure', fontdict=font_dict_title)

        axs[2][1].plot(Sg, 'b')
        axs[2][1].set_xlabel('x [m]', font_dict_axes)
        axs[2][1].set_ylabel('$s_g$ [-]', font_dict_axes)
        axs[2][1].set_title('Gas saturation', fontdict=font_dict_title)

        axs[2][2].plot(1 - Ss, 'b')
        axs[2][2].set_xlabel('x [m]', font_dict_axes)
        axs[2][2].set_ylabel('$\phi$ [-]', font_dict_axes)
        axs[2][2].set_title('Porosity', fontdict=font_dict_title)

        left = 0.05  # the left side of the subplots of the figure
        right = 0.95  # the right side of the subplots of the figure
        bottom = 0.05  # the bottom of the subplots of the figure
        top = 0.95  # the top of the subplots of the figure
        wspace = 0.25  # the amount of width reserved for blank space between subplots
        hspace = 0.25  # the amount of height reserved for white space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        for ii in range(3):
            for jj in range(3):
                for tick in axs[ii][jj].xaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

                for tick in axs[ii][jj].yaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

        plt.tight_layout()
        plt.savefig("results_kinetic_brief.pdf")
        plt.show()


class model_properties(property_container):
    def __init__(self, phases_name, components_name, elements_name, reaktoro, E_mat, Mw, min_z=1e-12,
                 diff_coef=float(0), rock_comp=1e-6, solid_dens=None):
        if solid_dens is None:
            solid_dens = []
        # Call base class constructor
        super().__init__(phases_name, components_name, Mw, min_z=min_z, diff_coef=diff_coef,
                         rock_comp=rock_comp, solid_dens=solid_dens)
        self.n_e = len(elements_name)
        self.elements_name = elements_name
        self.reaktoro = reaktoro
        self.E_mat = E_mat

    def run_flash(self, pressure, ze):
        # Make every value that is the min_z equal to 0, as Reaktoro can work with 0, but not transport
        ze = comp_extension(ze, self.min_z)
        self.nu, self.x, zc, density = Flash_Reaktoro(ze, 320, pressure, self.reaktoro)
        zc = comp_correction(zc, self.min_z)

        # Solid phase always needs to be present
        ph = list(range(len(self.nu)))  # ph = range(number of total phases)
        # min_ph = 0.01  # min value to be considered inside the phase
        # for i in range(len(self.nu)):
        #     if density[i] < 0 and min_ph < 0.1:
        #         min_ph_new = self.nu[i]
        #         min_ph = max(min_ph, min_ph_new)
        #     if density[1] > 1500 and min_ph < 0.1:
        #         min_ph_new = self.nu[1]
        #         min_ph = max(min_ph, min_ph_new)
        # if self.nu[0] <= min_ph:  # if vapor phase is less than min_z, it does not exist
        #     del ph[0]  # remove entry of vap
        #     density[0] = 0
        # elif self.nu[1] <= min_ph:  # if liq phase is less than min_z, it does not exist
        #     del ph[1]
        #     density[1] = 0
        # solid phase always present
        # for i in range(len(self.nu)):
        #     if i > len(self.nu)-1 and self.nu[i] < self.min_z:
        #         self.nu[i] = self.min_z
        # for i in range(len(self.nu)):
        #     if density[i] < 0 or density[1] > 2000:
        #         print('Partial molar problems, likely, density is below 0 or above 2000 for aqueous phase')
        #         print('ze', ze)
        #         print('zc', zc)
        #         print('nu', self.nu)
        #         print(density)
        return ph, zc, density


def comp_extension(z, min_z):
    sum_z = 0
    z_correct = False
    C = len(z)
    for c in range(C):
        new_z = z[c]
        if new_z <= min_z:
            new_z = 0
            z_correct = True
        elif new_z >= 1 - min_z:
            new_z = 1
            z_correct = True
        sum_z += new_z
    new_z = 1 - sum_z
    if new_z <= min_z:
        new_z = 0
        z_correct = True
    sum_z += new_z
    if z_correct:
        for c in range(C):
            new_z = z[c]
            if new_z <= min_z:
                new_z = 0
            elif new_z >= 1 - min_z:
                new_z = 1
            new_z = new_z / sum_z  # Rescale
            z[c] = new_z
    return z


def comp_correction(z, min_z):
    sum_z = 0
    z_correct = False
    C = len(z)
    for c in range(C):
        new_z = z[c]
        if new_z < min_z:
            new_z = min_z
            z_correct = True
        elif new_z > 1 - min_z:
            new_z = 1 - min_z
            z_correct = True
        sum_z += new_z  # sum previous z of the loop
    new_z = 1 - sum_z  # Get z_final
    if new_z < min_z:
        new_z = min_z
        z_correct = True
    sum_z += new_z  # Total sum of all z's
    if z_correct:  # if correction is needed
        for c in range(C):
            new_z = z[c]
            new_z = max(min_z, new_z)
            new_z = min(1 - min_z, new_z)  # Check whether z is in between min_z and 1-min_z
            new_z = new_z / sum_z  # Rescale
            z[c] = new_z
    return z


def Flash_Reaktoro(z_e, T, P, reaktoro):
    # if z_e[2] != z_e[3]:
    #     ze_new = (z_e[2]+z_e[3])/2
    #     z_e[2] = ze_new
    #     z_e[3] = ze_new
    #     # z_e = [float(i) / sum(z_e) for i in z_e]
    # if z_e[0] != z_e[1]:
    #     ze_new = (z_e[1] + z_e[0]) / 2
    #     z_e[0] = ze_new
    #     z_e[1] = ze_new
    reaktoro.addingproblem(T, P, z_e)
    nu, x, z_c, density = reaktoro.output()  # z_c order is determined by user, check if its the same order as E_mat
    return nu, x, z_c, density


class Reaktoro:
    def __init__(self):
        db = SupcrtDatabase("supcrtbl")
        db = PhreeqcDatabase('phreeqc.dat')
        # db = PhreeqcDatabase.fromFile("phreeqc_cat_ion.dat")

        '''Hardcode'''
        # 'H2O', 'Na+', 'Cl-', 'CO3-2', 'H+', 'OH-', 'Al+3', 'HCO3-', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-', 'Kaolinite', 'Quartz'
        # self.aq_comp = StringList(['Na+', 'Cl-', 'CO3-2', 'H+', 'OH-', 'Al+3', 'H3SiO4-', 'H2O', 'HCO3-', 'Al(OH)4-', 'H4SiO4'])
        self.aq_comp = StringList(['OH-', 'H+', 'Na+', 'Cl-', 'H2O'])
        # self.sol_comp = ['Kaolinite', 'Quartz']
        aq = AqueousPhase(self.aq_comp)
        # for i in range(len(self.sol_comp)):
        #     globals()['sol%s' % i] = MineralPhase(self.sol_comp[i])
        self.system = ChemicalSystem(db, aq)
        self.specs = EquilibriumSpecs(self.system)
        self.specs.temperature()
        self.specs.pressure()
        # self.specs.charge()
        # self.specs.openTo("Cl-")
        self.solver = EquilibriumSolver(self.specs)
        self.cp = type(object)
        # self.specs.pH()

    def addingproblem(self, temp, pres, z_e):
        state = ChemicalState(self.system)
        state.temperature(temp, 'kelvin')
        state.pressure(pres, 'bar')
        ne = self.aq_comp.size() - 2
        for i in range(ne):
            state.set(self.aq_comp[i], z_e[i], 'mol')
        # state.set('Kaolinite', z_e[i+1], 'mol')
        # state.set('Quartz', z_e[i + 1], 'mol')
        conditions = EquilibriumConditions(self.specs)
        conditions.temperature(temp, "celsius")
        conditions.pressure(pres, "bar")
        # conditions.charge(0)
        result = self.solver.solve(state, conditions)
        self.cp: ChemicalProps = ChemicalProps(state)
        self.failure = False
        if not result.optima.succeeded:
            print('Reaktoro did not find solution')
            self.failure = True
            print('z_e', z_e)
            # print(state)

    def output(self):
        # gas_props: ChemicalPropsPhaseConstRef = self.cp.phaseProps(0)
        liq_props: ChemicalPropsPhaseConstRef = self.cp.phaseProps(0)
        # sol_props: ChemicalPropsPhaseConstRef = self.cp.phaseProps(1)
        # sol_props2: ChemicalPropsPhaseConstRef = self.cp.phaseProps(2)
        aprops = AqueousProps(self.cp)

        '''Hardcode'''
        Na = self.cp.speciesAmount('Na+')
        Cl = self.cp.speciesAmount('Cl-')
        # CO3 = self.cp.speciesAmount('CO3-')
        H = self.cp.speciesAmount('H+')
        OH = self.cp.speciesAmount('OH-')
        # Al = self.cp.speciesAmount('Al+3')
        # H3SiO4 = self.cp.speciesAmount('H3SiO4-')
        # solid = self.cp.speciesAmount('Kaolinite')
        # solid2 = self.cp.speciesAmount('Quartz')
        H2O_aq = self.cp.speciesAmount('H2O')
        H2O = H2O_aq  # + H2O_g
        # HCO3 = self.cp.speciesAmount('HCO3-')
        # AlOH4 = self.cp.speciesAmount('Al(OH)4-')
        # H4SiO4 =  self.cp.speciesAmount('H4SiO4')
        # NaOH = self.cp.speciesAmount('NaOH')

        total_mol = self.cp.amount()
        total_mol_aq = liq_props.amount()
        #total_mol_sol = sol_props.amount()+sol_props2.amount()

        # mol_frac_gas_var = gas_props.speciesMoleFractions()
        mol_frac_aq_var = liq_props.speciesMoleFractions()

        '''Hardcode'''
        # mol_frac = [float(H2O/total_mol), float(Ca/total_mol), float(Na/total_mol),
        #             float(Cl/total_mol), float(X/total_mol), float(Nax/total_mol), float(Cax/total_mol)]
        # mol_frac_gas = [float(mol_frac_gas_var[0]), float(mol_frac_gas_var[1]), 0, 0, 0, 0, 0, 0]
        mol_frac_aq = [float(mol_frac_aq_var[0]), float(mol_frac_aq_var[1]), float(mol_frac_aq_var[2]),
                       float(mol_frac_aq_var[3]), float(mol_frac_aq_var[4])] #,
        #               float(mol_fraq_aq_var[6]), float(mol_fraq_aq_var[7]), float(mol_fraq_aq_var[8]),
        #               float(mol_fraq_aq_var[9]), float(mol_fraq_aq_var[10]), 0 ,0]
        #mol_frac_sol = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(solid/total_mol_sol), float(solid2/total_mol_sol)]

        # Partial molar volume equation: V_tot = total_mol * sum(molar_frac*partial mole volume)
        # partial_mol_vol_aq = np.zeros(len(mol_frac_aq))
        # for i in range(len(mol_frac_aq)-3):
        #     partial_mol_vol_aq[i] = float(liq_props.speciesStandardVolumes()[i])
        # partial_mol_vol_aq[4] = float(ion_props.speciesStandardVolumes()[0])
        # volume_aq = total_mol_aq * np.sum(np.multiply(mol_frac_aq, partial_mol_vol_aq))
        # print(float(ion_props.speciesStandardVolumes()[0]))
        # volume_gas = gas_props.volume()
        volume_aq = liq_props.volume()
        #volume_ion = 1e-5 * (total_mol_sol/total_mol)
        # volume_ion = (1/10)*volume_aq
        #volume_solid = sol_props.volume() + sol_props2.volume()
        volume_tot = self.cp.volume()

        # mass_aq = liq_props.mass() + self.cp.speciesMass('X-')
        # mass_ion = self.cp.speciesMass('NaX') + self.cp.speciesMass('CaX2')
        # density_aq = mass_aq/volume_aq
        # density_ion = mass_ion/volume_ion
        # density_gas = gas_props.density()
        density_aq = liq_props.density()
        #density_ion = 2700
        # density = self.cp.density()
        #density_solid = (2 * sol_props.density() * sol_props2.density()) / (sol_props.density()+sol_props2.density())

        # S_g = volume_gas / volume_tot
        S_w = volume_aq / volume_tot
        # S_ion = volume_ion / volume_tot
        #S_ion = volume_ion / volume_tot
        #S_s = volume_solid / volume_tot

        # V = (density_gas * S_g) / (density_gas * S_g + density_aq * S_w + density_solid * S_s)
        # L = (density_aq * S_w) / (density_gas * S_g + density_aq * S_w + density_solid * S_s)
        #L = (density_aq * S_w) / (density_solid * S_s + density_aq * S_w)
        #S = (density_solid * S_s) / (density_aq * S_w + density_solid * S_s)
        # nu = [float(V), float(L), float(S)]
        nu = [1]#[float(L), float(S)]#, float(ion)]
        # x = [mol_frac_gas, mol_frac_aq, mol_frac_sol]
        x = mol_frac_aq#, mol_frac_sol]
        '''Hardcode'''
        # z_c = [float(H2O / total_mol), float(CO2 / total_mol),
        #        float(Na / total_mol), float(Cl / total_mol), float(Na2 / total_mol), float(Cl2 / total_mol),
        #        float(solid / total_mol), float(solid2 / total_mol)]
        #        'Na+', 'Cl-', 'CO3-2', 'H+', 'OH-', 'Al+3', 'H3SiO4-', 'H2O', 'HCO3-', 'Al(OH)4-', 'H4SiO4'

        z_c = [float(OH / total_mol), float(H / total_mol), float(Na / total_mol), float(Cl / total_mol),
               float(H2O/total_mol)]
# 'Na+', 'Cl-', 'CO3-2', 'H+', 'OH-', 'Al+3', 'H3SiO4-'
# 'H2O', 'Na+', 'Cl-', 'CO3-2', 'H+', 'OH-', 'Al+3', 'HCO3-', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-', 'Kaolinite', 'Quartz'
        # density = [float(density_gas), float(density_aq), float(density_solid)]
        # if density < 0 or density > 1500:
        #     print(z_c)
        #     print(density)
        density = [density_aq]  # , density_solid]  #, density_ion]

        if self.failure:
            print('z_c', z_c)
        return nu, x, z_c, density