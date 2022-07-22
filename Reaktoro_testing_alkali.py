from reaktoro import *
import numpy as np

# db = PhreeqcDatabase.fromFile('phreeqc_cat_ion.dat')
db = PhreeqcDatabase('phreeqc.dat')

# aq = AqueousPhase(StringList(['H2O', 'Na+', 'Cl-', 'CO3-2', 'H+', 'OH-', 'HCO3-', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-']))
# aq = AqueousPhase(StringList(['H2O', 'Na+', 'Cl-', 'CO3-2', 'H+', 'OH-', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-']))
# aq = AqueousPhase(speciate(StringList("H O C Na Cl Al Si")))
# ol = MineralPhase('Kaolinite')
#sol2 = MineralPhase('Quartz')
aq = AqueousPhase(StringList(['H2O', 'H+', 'Na+', 'Cl-', 'OH-', 'CO2']))
system = ChemicalSystem(db, aq)
state = ChemicalState(system)
specs = EquilibriumSpecs(system)
specs.temperature()
specs.pressure()
#specs.pH()
specs.charge()
specs.openTo('Cl-')
solver = EquilibriumSolver(specs)
conditions = EquilibriumConditions(specs)
conditions.temperature(320, 'kelvin')
conditions.pressure(200, 'bar')
#conditions.pH(7)
conditions.charge(0)
state.setTemperature(320, "kelvin")
state.setPressure(200, "bar")
# H2O, Ca, Na, Cl, X
# [0.91 0.02 0.02 0.05 0.  ]
# z_e [0.93 0.02 0.   0.05 0.  ]
# z_e = [0.87, 0.02, 0.01, 0.05, 0.05]
z_e = [1, 1, 0, 0]#, 0.001, 0.01, 0.001, 0.001]

state.set("H2O", z_e[0], "kg")
state.set('Na+', z_e[1], 'mol')
state.set('Cl-', z_e[2], 'mol')
state.set('CO2', z_e[3], 'mol')
# state.set("Cl-", z_e[2], "mol")
# state.set("CO3-2", z_e[3], "mol")
# state.set('Kaolinite', z_e[4], "mol")
#state.set('Quartz', z_e[5], 'mol')


#print(aprops.pH())
result = solver.solve(state, conditions)
if not result.optima.succeeded:
        print('Failed')
#print(state)
cp = ChemicalProps(state)
aprops = AqueousProps(cp)
print(aprops)
print(state)

