from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pyscipopt
import os

# provide an email address
os.environ['NEOS_EMAIL'] = 'a30785@alunos.ipca.pt'

# Define the problem data
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
}

machines = [1, 2, 3, 4, 5, 6, 7, 8]

# Model definition
model = ConcreteModel()

# Sets
model.J = Set(initialize=jobs.keys(), doc="Jobs")
model.M = Set(initialize=machines, doc="Machines")
model.O = Set(initialize=[(job, operation) for job in jobs for operation in range(len(jobs[job]))], doc="Operations")

# Parameters
processing_time = {(job, op, m): t for job in jobs for op, times in enumerate(jobs[job]) for m, t in times}

# Decision Variables
model.start_time = Var(model.O, within=NonNegativeReals, doc="Start time of each operation")
model.assignment = Var(model.O, model.M, within=Binary, doc="Assignment of operations to machines")
#model.makespan = Var(within=NonNegativeReals, doc="Makespan of the schedule")
model.makespan = Var(within=Integers, doc="Makespan of the schedule")

# Constraints

# 1-Each operation must be assigned to exactly one machine
def one_machine_rule(model, j, o):
    return sum(model.assignment[j, o, m] for m in model.M if (j, o, m) in processing_time) == 1
model.One_Machine_Rule = Constraint(model.O, rule=one_machine_rule)

# 2-Precedence constraint: Each operation must start after the previous one finishes
def precedence_rule(model, j, o):
    if o > 0:
        prev_op = o - 1
        return model.start_time[j, o] >= model.start_time[j, prev_op] + sum(
            model.assignment[j, prev_op, m] * processing_time[j, prev_op, m] for m in model.M if (j, prev_op, m) in processing_time
        )
    return Constraint.Skip
model.Precedence_Rule = Constraint(model.O, rule=precedence_rule)

# Big-M value (a sufficiently large number)
# Calculate a better Big-M value based on problem characteristics
max_processing_time = max(t for job in jobs.values() 
                         for op in job 
                         for _, t in op)
total_operations = sum(len(ops) for ops in jobs.values())
M = max_processing_time * total_operations  # More realistic Big-M

# New binary variable for tracking precedence
model.precedence = Var(model.O, model.O, within=Binary, doc="Binary variable to track precedence between operations")

def precedence_no_overlap_rule(model, j1, o1, j2, o2, m):
    if (j1, o1) < (j2, o2) and j1 != j2 and (j1, o1, m) in processing_time and (j2, o2, m) in processing_time:

        # Prevent overlap if different jobs assigned to the same machine
        return model.start_time[j2, o2] >= model.start_time[j1, o1] + processing_time[j1, o1, m] - (3 - model.precedence[j1, o1, j2, o2] - model.assignment[j1, o1, m] - model.assignment[j2, o2, m] ) * M
    return Constraint.Skip

def precedence_no_overlap_no_precedence_rule(model, j1, o1, j2, o2, m):
    if (j1, o1) < (j2, o2) and j1 != j2 and (j1, o1, m) in processing_time and (j2, o2, m) in processing_time:

        # Prevent overlap if different jobs assigned to the same machine
        return model.start_time[j1, o1] >= model.start_time[j2, o2] + processing_time[j2, o2, m] - (2 + model.precedence[j1, o1, j2, o2] - model.assignment[j1, o1, m] - model.assignment[j2, o2, m] ) * M
    return Constraint.Skip

# 3- Add constraint no Overlap with Precedence and without Precedence
model.Precedence_No_Overlap = Constraint(model.O, model.O, model.M, rule=precedence_no_overlap_rule)
model.Precedence_No_OverlapWithoutPrecedences = Constraint(model.O, model.O, model.M, rule=precedence_no_overlap_no_precedence_rule)

# 4. Makespan definition
def makespan_rule(model, j, o):
    return model.makespan >= model.start_time[j, o] + sum(
        model.assignment[j, o, m] * processing_time[j, o, m] for m in model.M if (j, o, m) in processing_time
    )
model.Makespan_Constraint = Constraint(model.O, rule=makespan_rule)

# Print the model
def print_schedule_results(model, jobs):
    """Print detailed schedule results in a formatted way"""
    # Check if we have a valid solution
    try:
        makespan = value(model.makespan)
        print("\n" + "="*50)
        print("SCHEDULE RESULTS")
        print("="*50)
        print(f"Optimal Makespan: {makespan:.2f} time units")
        print("-"*50)
        
         # Sort operations by start time for clearer output
        schedule = []
        for j in model.J:
            for op in range(len(jobs[j])):
                for m, _ in jobs[j][op]:  # Only iterate over valid machines for each operation
                    if hasattr(model.assignment[j, op, m], 'value') and \
                       value(model.assignment[j, op, m]) is not None and \
                       value(model.assignment[j, op, m]) > 0.5:
                        schedule.append({
                            'job': j,
                            'operation': op + 1,
                            'machine': m,
                            'start': value(model.start_time[j, op]),
                            'duration': processing_time[j, op, m],
                            'end': value(model.start_time[j, op]) + processing_time[j, op, m]
                        })
        
        if not schedule:
            print("No operations were scheduled.")
            return
            
        # Sort by start time
        schedule.sort(key=lambda x: x['start'])
        
        # Print schedule in a tabulated format
        print("\nDetailed Schedule:")
        print(f"{'Job':^5} {'Op':^5} {'Machine':^8} {'Start':^10} {'End':^10} {'Duration':^10}")
        print("-"*50)
        for item in schedule:
            print(f"{item['job']:^5} {item['operation']:^5} {item['machine']:^8} "
                  f"{item['start']:^10.2f} {item['end']:^10.2f} {item['duration']:^10.2f}")
        
        # Print machine utilization
        print("\nMachine Utilization:")
        print("-"*50)
        for m in model.M:
            machine_ops = [s for s in schedule if s['machine'] == m]
            total_time = sum(op['duration'] for op in machine_ops)
            utilization = (total_time / makespan) * 100 if makespan > 0 else 0
            print(f"Machine {m}: {utilization:.1f}% utilized")
    
    except Exception as e:
        print(f"Error generating schedule results: {str(e)}")

# Plotting function
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
    return fig

# Solvers to try
solvers = ['glpk', 'gurobi', 'ipopt', 'neos']  # Add or remove solvers as needed
solver_num = 0
solver_name = solvers[solver_num]  # Change this to try different solvers
# Objective Function
model.objective = Objective(expr=model.makespan, sense=minimize)

if solver_num != 3:
    try:
        solver = SolverFactory(solver_name)

        if solver is None:  # Check if the solver is available
            print(f"Solver '{solver_name}' not found. Skipping.")

        if solver_num == 0:  # GLPK
            results = solver.solve(model, tee=True, options={'mipgap': 0.00001, 'tmlim': 900}) 
            # Note the tmlim, not timelimit.
            # mipgap option is essential for controlling the trade-off between solution quality and solution time. It allows you to get good, near-optimal solutions within a reasonable time frame.
        else: 
            results = solver.solve(model, tee=True)  # tee=True for solver output

        if results.solver.status == SolverStatus.ok:
            print(f"Solution found using {solver_name}!")
            print_schedule_results(model, jobs)  # Your existing function
            fig = plot_schedule(model, processing_time, machines, jobs)
            if fig:
                plt.show()
                # Optionally save the plot with a solver-specific name:
                # fig.savefig(f"schedule_{solver_name}.png", bbox_inches="tight")
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            print(f"Model is infeasible with {solver_name}.")
        elif results.solver.termination_condition == TerminationCondition.unbounded:
            print(f"Model is unbounded with {solver_name}.")
        else:
            print(f"Solver {solver_name} did not find an optimal solution. Status: {results.solver.termination_condition}")

    except Exception as e:
        print(f"An error occurred with solver {solver_name}: {e}")
