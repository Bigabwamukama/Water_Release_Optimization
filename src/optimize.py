import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
import matplotlib.pyplot as plt
import constants as cts
from models import Model
import math

plt.rcParams.update({'font.size': 13})

model = Model(1)

class DischargeOptimization(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=1,  # One decision variable (water level)
                         n_obj=2,  # Two objectives
                         n_constr=2,  # One constraint
                         xl=np.array([10.5]),  # Lower bound for x
                         xu=np.array([13.77]))  # Upper bound for x

    def _evaluate(self, x, out, *args, **kwargs):
        Q = model.predict(x[0])[0]  # Compute discharge        
        x_normal = 12  # Target normal water level        
        # Objective 1: Maximize Q (flood control) → minimize -Q
        obj1 = -Q  
        # Objective 2: Keep water level near normal
        obj2 = max(0, (x[0] - x_normal)) 
        # Store objectives as a numpy array
        out["F"] = np.array([obj1, obj2])
        # Constraints:
        # 1. Discharge should be below 4200 (Q - 4200 ≤ 0)
        g1 = Q - 4200
        # 2. Water level should be between 11 and 13.5 (handled as constraints)
        g2 = max(0, 11 - x[0]) + max(0, x[0] - 13.5)  # Ensures water level stays in range
        out["G"] = np.array([g1, g2])

# Define NSGA-II optimization algorithm
algorithm = NSGA2(
    pop_size=100,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=20),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# Run optimization
res = minimize(
    DischargeOptimization(),
    algorithm,
    termination=get_termination("n_gen", 100),
    verbose=True
)

i = 0
wls = []
Qs = []
for i in range (0,len(res.X)):
    wls.append(res.X[i][0])
    Qs.append(model.predict(res.X[i])[0])
pareto_solutions_df = pd.DataFrame({'Water Level': wls, f'Discharge ($m^3$/s)': Qs})
pareto_solutions_df.to_csv(cts.PARETALO_SOLN, index=False)
F = res.F 
F_sorted = F[np.argsort(F[:, 0])]
pareto_front_df = pd.DataFrame({f'-Discharge ($m^3$/s)': F_sorted[:, 0], 
                                'Deviation from Normal Water Level': F_sorted[:, 1]})
pareto_front_df.to_csv(cts.PARETALO_FRONT, index=False)
plt.figure(figsize=(8, 6))
plt.plot(F_sorted[:, 0], F_sorted[:, 1],'b', label='Pareto Front')
plt.xlabel(f"-Discharge ($m^3$/s)")
plt.ylabel("Deviation from Normal Water Level(m)")
plt.title("Pareto Front: Flood Control vs. Water Level Stability")
plt.legend()
plt.grid()
plt.savefig(cts.FRONT_PLOT_NEGATIVE, bbox_inches='tight')
plt.show()
F = res.F  
for i in range(len(F[:, 0])) :
    F[:, 0][i] = -F[:, 0][i]
F_sorted = F[np.argsort(F[:, 0])]
plt.figure(figsize=(8, 6))
plt.plot(F_sorted[:, 0], F_sorted[:, 1],'b', label='Pareto Front') 
plt.xlabel(f"Discharge ($m^3$/s)") 
plt.ylabel("Deviation from Normal Water Level(m)")
plt.legend()
plt.grid()
plt.savefig(cts.FRONT_PLOT_POSITIVE, bbox_inches='tight')
plt.show()
a = 66.3   
b = 2.01     
Ho = 7.96   
wl_optimized = np.array([x[0] for x in res.X])
wl_optimized = np.sort(wl_optimized)
Q_optimized = np.array([model.predict(wl)[0] for wl in wl_optimized])
Q_rating = np.array([a * (wl - Ho)**b for wl in wl_optimized])
F = res.F 
F_sorted = F[np.argsort(F[:, 0])]
deviations = F_sorted[:, 1]
for i in range(len(F_sorted[:, 0])) :
    F_sorted[:, 0][i] = F_sorted[:, 0][i]
Q_optimized_real = F_sorted[:, 0]
plt.figure(figsize=(10, 6))
plt.plot(wl_optimized, Q_optimized, 'b', label='Optimized Discharge')
plt.plot(wl_optimized, Q_rating, 'r', label='Rating Curve Discharge')
plt.xlabel("Water Level (m)")
plt.ylabel("Discharge (m³/s)")
plt.title("Water Level vs. Discharge: Optimized vs Rating Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(cts.DISCHARGE_AGREED_WL_COMPARISON_PLOT, bbox_inches='tight')
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(Q_optimized_real, deviations, 'b', label='Optimized Q vs Deviation')
plt.plot(Q_rating, deviations, 'r', label='Rating Curve Q vs Same Deviation')
plt.xlabel("Discharge (m³/s)")
plt.ylabel("Deviation from Normal Water Level (m)")
plt.title("Deviation vs. Discharge: Optimized vs Rating Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(cts.DEVIATION_DISCHARGED_AGREED_COMPARISON_PLOT, bbox_inches='tight')
plt.show()
compare_df = pd.DataFrame({"Water Level":wl_optimized,
                           "Optimized_Discharge":Q_optimized,
                           "Rating Curve Discharge":Q_rating, 
                           "Deviation from Normal WL": deviations})
compare_df.to_csv(cts.PARETALO_AGREED,index=False)




