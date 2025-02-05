from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class JobShopData:
    """Data structure to hold job shop scheduling data"""
    jobs: Dict[int, List[List[Tuple[int, int]]]]
    machines: List[int]

class JobShopScheduler:
    def __init__(self, data: JobShopData):
        self.data = data
        self.model = None
        self.processing_time = self._create_processing_time()
        
    def _create_processing_time(self) -> Dict[Tuple[int, int, int], int]:
        """Create processing time dictionary with better memory efficiency"""
        return {(job, op, m): t 
                for job in self.data.jobs 
                for op, times in enumerate(self.data.jobs[job]) 
                for m, t in times}

    def create_model(self) -> ConcreteModel:
        """Create and return the optimization model with improved constraints"""
        model = ConcreteModel()
        
        # Sets
        model.J = Set(initialize=self.data.jobs.keys())
        model.M = Set(initialize=self.data.machines)
        model.O = Set(initialize=[(j, o) for j in self.data.jobs 
                                for o in range(len(self.data.jobs[j]))])
        
        # Variables
        model.start_time = Var(model.O, within=NonNegativeReals)
        model.assignment = Var(model.O, model.M, within=Binary)
        model.makespan = Var(within=NonNegativeReals)
        
        # Calculate max processing time for better Big-M value
        max_processing_time = max(self.processing_time.values())
        M = len(self.data.jobs) * len(self.data.machines) * max_processing_time
        
        # Improved constraints
        def one_machine_rule(m, j, o):
            return sum(m.assignment[j, o, machine] 
                      for machine in m.M 
                      if (j, o, machine) in self.processing_time) == 1
        model.One_Machine_Rule = Constraint(model.O, rule=one_machine_rule)
        
        def precedence_rule(m, j, o):
            if o > 0:
                prev_op = o - 1
                return m.start_time[j, o] >= m.start_time[j, prev_op] + \
                       sum(m.assignment[j, prev_op, machine] * self.processing_time[j, prev_op, machine] 
                           for machine in m.M if (j, prev_op, machine) in self.processing_time)
            return Constraint.Skip
        model.Precedence_Rule = Constraint(model.O, rule=precedence_rule)
        
        def no_overlap_rule(m, j1, o1, j2, o2, machine):
            if ((j1, o1) < (j2, o2) and j1 != j2 and 
                (j1, o1, machine) in self.processing_time and 
                (j2, o2, machine) in self.processing_time):
                return m.start_time[j2, o2] >= m.start_time[j1, o1] + \
                       self.processing_time[j1, o1, machine] - \
                       M * (2 - m.assignment[j1, o1, machine] - m.assignment[j2, o2, machine])
            return Constraint.Skip
        model.No_Overlap = Constraint(model.O, model.O, model.M, rule=no_overlap_rule)
        
        def makespan_rule(m, j, o):
            return m.makespan >= m.start_time[j, o] + \
                   sum(m.assignment[j, o, machine] * self.processing_time[j, o, machine] 
                       for machine in m.M if (j, o, machine) in self.processing_time)
        model.Makespan_Rule = Constraint(model.O, rule=makespan_rule)
        
        # Objective
        model.objective = Objective(expr=model.makespan, sense=minimize)
        
        self.model = model
        return model

    def solve(self, solver_name) -> bool:
        """Solve the model with specified solver"""
        if self.model is None:
            self.create_model()
            
        solver = SolverFactory(solver_name)
        results = solver.solve(self.model)
        return results.solver.status == SolverStatus.ok

    def plot_schedule(model, processing_time, machines, jobs):
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.get_cmap("tab10")
        job_colors = {}
        patches = {}

        for j in model.J:
            for op in range(len(jobs[j])):
                for m in model.M:
                    if model.assignment[j, op, m]() == 1:
                        if j not in job_colors:
                            job_colors[j] = colors(j % 10)
                        color = job_colors[j]

                        start = model.start_time[j, op]()  # Get the start time
                        duration = processing_time[j, op, m]  # Get the duration

                        # Draw the barh using start and duration!
                        ax.barh(m, duration, left=start, color=color, edgecolor='black')  # <--- This was missing!

                        ax.text(start + duration / 2, m, f'O{op+1}', va='center', ha='center',
                                color='white', fontsize=10, fontweight='bold')

                        if j not in patches:
                            patches[j] = mpatches.Patch(color=color, label=f'Job {j}')

                        break  # Break after finding the assigned machine

        ax.set_xlabel("Time Units")
        ax.set_ylabel("Machines")
        ax.set_title("Job-Shop Schedule Gantt Chart")
        ax.set_yticks(machines)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Correct legend creation:
        handles = [patches[j] for j in sorted(patches.keys())]
        ax.legend(handles=handles,
                title="Jobs",
                loc='upper right',
                fontsize='medium',
                handlelength=1.5,
                handletextpad=0.5,
                bbox_to_anchor=(1.15, 1),
                borderaxespad=0.
                )

        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    data = JobShopData(
        jobs = {
                    1: [
                        [(1, 4), (3, 5)],              				# Operation 1: Machines 1 and 3, with times 3 and 5
                        [(2, 4), (4, 5)],              				# Operation 2: Machines 2 and 4, with times 4 and 5
                        [(3, 5), (5, 6)],              				# Operation 3: Machines 3 and 5, with times 5 and 6
                        [(4, 5), (5, 5), (6, 4), (7, 5), (8, 9)]    # Operation 4: Machines 4, 5, 6, 7, and 8, with times 5, 5, 4, 5, and 9            
                    ],
                    2: [
                        [(1, 1), (3, 5), (5, 7)],     				# Operation 1: Machines 1, 3, and 5, with times 1, 5, and 7
                        [(4, 5), (8, 4)],             				# Operation 2: Machines 4 and 8, with times 5 and 4
                        [(4, 1), (6, 6)],             				# Operation 3: Machines 4 and 6, with times 1 and 6
                        [(4, 4), (7, 4), (8, 7)],     				# Operation 4: Machines 4, 7, and 8, with times 4, 4, and 7
                        [(4, 1), (6, 2)],             				# Operation 5: Machines 4 and 6, with times 1 and 2
                        [(1, 5), (6, 6), (8, 4)],     				# Operation 6:  Machines 1, 6, and 8, with times 5, 6, and 4
                        [(4, 4)]                      				# Operation 7: Machine 4, with time 4
                    ],                                				
                    3: [                              				
                        [(2, 7), (3, 6), (8, 8)],     				# Operation 1: Machines 2, 3, and 8, with times 7, 6, and 8
                        [(4, 7), (8, 7)],             				# Operation 2: Machines 4 and 8, with times 7 and 7
                        [(3, 7), (5, 8), (7, 7)],     				# Operation 3: Machines 3, 5, and 7, with times 7, 8, and 7
                        [(4, 7), (6, 8)],             				# Operation 4: Machines 4 and 6, with times 7 and 8
                        [(1, 1), (2, 4)]             				# Operation 5 : Machines 1 and 2, with times 1 and 4  
                    ],                                
                    4: [                              
                        [(1, 4), (3, 3), (5, 7)],     				# Operation 1: Machines 1, 3, and 5, with times 4, 3, and 7
                        [(2, 4), (8, 4)],             				# Operation 2: Machines 2 and 8, with times 4 and 4
                        [(3, 4), (4, 5), (6, 6), (7, 7)],  			# Operation 3: Machines 3, 4, 6, and 7, with times 4, 5, 6, and 7
                        [(5, 3), (6, 5), (8, 5)]     				# Operation 4   : Machines 5, 6, and 8, with times 3, 5, and 5
                    ],                                
                    5: [                              
                        [(1, 3)],                     				# Operation 1: Machine 1, with time 3
                        [(2, 4), (4, 5)],             				# Operation 2: Machines 2 and 4, with times 4 and 5
                        [(3, 4), (8, 4)],             				# Operation 3: Machines 3 and 8, with times 4 and 4
                        [(5, 3), (6, 3), (8, 3)],     				# Operation 4: Machines 5, 6, and 8, with times 3, 3, and 3
                        [(4, 5), (6, 4)]             				# Operation 5 : Machines 4 and 6, with times 5 and 4  				
                    ],                                				
                    6: [                              				
                        [(1, 3), (2, 5), (3, 6)],     				# Operation 1: Machines 1, 2, and 3, with times 3, 5, and 6
                        [(4, 7), (5, 8)],             				# Operation 2: Machines 4 and 5, with times 7 and 8
                        [(3, 9), (6, 8)]             				# Operation 3 : Machines 3 and 6, with times 9 and 8     
                    ],                                
                    7: [                              
                        [(3, 4), (5, 5), (6, 4)],     				# Operation 1: Machines 3, 5, and 6, with times 4, 5, and 4
                        [(4, 4), (7, 6), (8, 4)],     				# Operation 2: Machines 4, 7, and 8, with times 4, 6, and 4
                        [(1, 3), (3, 3), (4, 4), (5, 5)],  			# Operation 3: Machines 1, 3, 4, and 5, with times 3, 3, 4, and 5
                        [(4, 4), (6, 6), (8, 5)],     				# Operation 4: Machines 4, 6, and 8, with times 4, 6, and 5
                        [(1, 3), (3, 3)]             				# Operation 5 : Machines 1 and 3, with times 3 and 3
                    ],                                
                    8: [                              
                        [(1, 3), (2, 4), (6, 4)],     				# Operation 1: Machines 1, 2, and 6, with times 3, 4, and 4
                        [(4, 6), (5, 5), (8, 4)],     				# Operation 2: Machines 4, 5, and 8, with times 6, 5, and 4
                        [(3, 4), (7, 5)],             				# Operation 3: Machines 3 and 7, with times 4 and 5
                        [(4, 4), (6, 6)],             				# Operation 4: Machines 4 and 6, with times 4 and 6
                        [(7, 1), (8, 2)]             				# Operation 5: Machines 7 and 8, with times 1 and 2
                    ]
                },
        machines=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    
    scheduler = JobShopScheduler(data)
    if scheduler.solve("gurobi"):
        print(f"Optimal Makespan: {scheduler.model.makespan()}")

    else:
        print("No solution found")

if __name__ == "__main__":
    main()