from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# Create the model
model = cp_model.CpModel()

# Variables
horizon = sum(max(t for m, t in op) for job in jobs.values() for op in job)
start_times = {}
assignments = {}
durations = {}
makespan = model.NewIntVar(0, horizon, 'makespan')  # Define makespan variable

# Create variables and constraints
for job_id, operations in jobs.items():
    for op_id, op in enumerate(operations):
        op_key = (job_id, op_id)
        start_times[op_key] = model.NewIntVar(0, horizon, f'start_{job_id}_{op_id}')
        assignments[op_key] = {}
        durations[op_key] = {}
        for machine, duration in op:
            assignments[op_key][machine] = model.NewBoolVar(f'assignment_{job_id}{op_id}{machine}')
            durations[op_key][machine] = duration

# Constraints
# Each operation must be assigned to exactly one machine
for op_key in assignments:
    model.Add(sum(assignments[op_key].values()) == 1)

# Precedence constraints
for job_id, operations in jobs.items():
    for op_id in range(1, len(operations)):
        prev_op_key = (job_id, op_id - 1)
        current_op_key = (job_id, op_id)
        model.Add(start_times[current_op_key] >= start_times[prev_op_key] + sum(
            assignments[prev_op_key][m] * durations[prev_op_key][m] for m in assignments[prev_op_key]
        ))

# No overlap constraints
for m in machines:
    intervals = []
    for op_key in assignments:
        if m in assignments[op_key]:
            duration = durations[op_key][m]
            interval = model.NewOptionalIntervalVar(
                        start_times[op_key], 
                        durations[op_key][m],  # Use fixed duration
                        start_times[op_key] + durations[op_key][m], 
                        assignments[op_key][m],  # This ensures the interval is used only when assigned
                        f'interval_{op_key}_{m}'
                    )
            intervals.append(interval)
    model.AddNoOverlap(intervals)

# *Makespan Constraint (Newly Added)*
for op_key in start_times:
    model.Add(makespan >= start_times[op_key] + sum(
        assignments[op_key][m] * durations[op_key][m] for m in assignments[op_key]
    ))

# Objective
model.Minimize(makespan)

class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    def _init_(self, variables):
        cp_model.CpSolverSolutionCallback._init_(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print(f"{v} = {self.Value(v)}")

# Attach the callback to the solver
class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables): # Changed _init_ to __init__
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print(f"{v} = {self.Value(v)}")

# Attach the callback to the solver
solution_printer = VarArraySolutionPrinter([start_times[op_key] for op_key in start_times])

# Solve the model
solver = cp_model.CpSolver()

solver.parameters.log_search_progress = True
solver.parameters.cp_model_probing_level = 0  # Reduces aggressive simplifications

# Run IIS computation
#This will generate an IIS report if the model is infeasible
status = solver.Solve(model,solution_printer)

if status == cp_model.OPTIMAL:
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray"]
    job_colors = {}
    
    for job_id in jobs.keys():
        job_colors[job_id] = colors[job_id % len(colors)]
    
    for job_id, operations in jobs.items():
        for op_id in range(len(operations)):
            op_key = (job_id, op_id)
            for m in assignments[op_key]:
                if solver.Value(assignments[op_key][m]):
                    start = solver.Value(start_times[op_key])
                    duration = durations[op_key][m]
                    ax.broken_barh([(start, duration)], (m - 0.4, 0.8), facecolors=job_colors[job_id])
                    ax.text(start + duration / 2, m, f"O{op_id + 1}", ha='center', va='center', color='white', fontsize=8)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_yticks(machines)
    ax.set_yticklabels([f"Machine {m}" for m in machines])
    ax.set_title("Job Shop Scheduling Gantt Chart")
    
    legend_patches = [mpatches.Patch(color=color, label=f"Job {job}") for job, color in job_colors.items()]
    ax.legend(handles=legend_patches, loc='upper right')
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
elif status == cp_model.FEASIBLE:
    print("Feasible solution found.")
elif status == cp_model.INFEASIBLE:
    print("The problem is infeasible.")
    # Try to get the minimal set of conflicting constraints

    assumptions = []
    for op_key in start_times:
        assumptions.append(start_times[op_key] >= 0)
    solver.SolveWithSolutionCallback(model, solution_printer)

    infeasible_assumptions = solver.SufficientAssumptionsForInfeasibility()
    if infeasible_assumptions:
        print("Minimal set of conflicting constraints:")
        for assumption in infeasible_assumptions:
            print(f"  {assumption}")
    else:
        print("No assumptions returned. The issue might be more complex.")

elif status == cp_model.MODEL_INVALID:
    print("The model is invalid.")
else:
    print("Unknown solver status.")

print("Jobs:", jobs)
print("Machines:", machines)
print("Horizon:", horizon)

# Print the results
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"Makespan: {solver.Value(makespan)}")
    for job_id, operations in jobs.items():
        for op_id in range(len(operations)):
            op_key = (job_id, op_id)
            for m in assignments[op_key]:
                if solver.Value(assignments[op_key][m]):
                    print(f"Job {job_id}, Operation {op_id + 1} on Machine {m} starts at {solver.Value(start_times[op_key])}")
else:
    print("No solution found.")
    