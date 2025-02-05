from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np

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

'''
Each element in model.O represents an operation in a specific job. The tuple (i, j) means:

i → Job number
j → Operation index within that job
Let's break down Job 1:

Job 1 Breakdown
In the jobs dictionary:

1: [
    [(1, 4), (3, 5)],                # Operation 0
    [(2, 4), (4, 5)],                # Operation 1
    [(3, 5), (5, 6)],                # Operation 2
    [(4, 5), (5, 5), (6, 4), (7, 5), (8, 9)]  # Operation 3
]
Now, let's match the (i, j) pairs in model.O:

Tuple (i, j)	Job ID	Operation Index	Machines & Processing Times
(1, 0)	1	0	Machines: 1 (4 units), 3 (5 units)
(1, 1)	1	1	Machines: 2 (4 units), 4 (5 units)
(1, 2)	1	2	Machines: 3 (5 units), 5 (6 units)
(1, 3)	1	3	Machines: 4 (5 units), 5 (5 units), 6 (4 units), 7 (5 units), 8 (9 units)

Interpretation
(1, 0) → First operation of Job 1, can be processed on Machine 1 (4 time units) or Machine 3 (5 time units).
(1, 1) → Second operation of Job 1, can be processed on Machine 2 (4 time units) or Machine 4 (5 time units).
(1, 2) → Third operation of Job 1, can be processed on Machine 3 (5 time units) or Machine 5 (6 time units).
(1, 3) → Fourth operation of Job 1, can be processed on multiple machines (4, 5, 6, 7, or 8) with different times.
These operations must be scheduled in sequence (i.e., (1, 0) must finish before (1, 1), etc.), and each operation can only be assigned to one machine from the given options.
'''
# Parameters
processing_time = {(i, j, m): t for i in jobs for j, ops in enumerate(jobs[i]) for m, t in ops}

# Decision Variables
model.start_time = Var(model.O, within=NonNegativeReals, doc="Start time of each operation")
model.assignment = Var(model.O, model.M, within=Binary, doc="Assignment of operations to machines")
model.makespan = Var(within=NonNegativeReals, doc="Makespan of the schedule")

# Constraints

# def precedence_rule(model, job, operation):
#     if operation < 1:
#         return Constraint.Skip
#     return model.start_time[job, operation] >= model.start_time[job, operation-1] + sum(model.assignment[job, operation-1, m] * processing_time[job, operation-1, m] for m in machines if (job, operation-1, m) in processing_time)
# model.precedence = Constraint(model.O, rule=precedence_rule)

# def machine_constraint_rule(model, i, j):
#     return sum(model.assignment[i, j, m] for m in machines if (i, j, m) in processing_time) == 1
# model.machine_assignment = Constraint(model.O, rule=machine_constraint_rule)

# def makespan_rule(model, i, j):
#     return model.makespan >= model.start_time[i, j] + sum(
#         model.assignment[i, j, m] * processing_time[i, j, m] for m in machines if (i, j, m) in processing_time
#     )
# model.makespan_constraint = Constraint(model.O, rule=makespan_rule)

def precedence_rule(model, job, operation):
    if operation < 1:
        return Constraint.Skip  # Skip the first operation as it has no predecessor
    return model.start_time[job, operation] >= model.start_time[job, operation-1] + sum(
        model.assignment[job, operation-1, m] * processing_time[job, operation-1, m] 
        for m in machines if (job, operation-1, m) in processing_time
    )

model.precedence = Constraint(model.O, rule=precedence_rule)

def machine_constraint_rule(model, i, j):
   return sum(model.assignment[i, j, m] for m in machines if (i, j, m) in processing_time) == 1

model.machine_assignment = Constraint(model.O, rule=machine_constraint_rule)


# Constraint: Assign exactly one machine to each operation (as before)
def assign_one_machine_rule(model, job, operation):
    return sum(model.assignment[job, operation, m] for m in model.M) == 1

#model.assign_one_machine = Constraint(model.O, rule=assign_one_machine_rule)

# Constraint: Relate start time, processing time, and makespan
def processing_time_constraint_rule(model, job, operation):
    # Calculate the processing time on the assigned machine
    processing_time_for_op = sum(model.assignment[job, operation, m] * processing_time.get((job, operation, m), 0) for m in model.M)

    # The start time + processing time must be less than or equal to the makespan.
    # This implicitly ensures the start time is correct.
    return model.start_time[job, operation] + processing_time_for_op <= model.makespan

#model.processing_time_constraint = Constraint(model.O, rule=processing_time_constraint_rule)


def makespan_rule(model, i, j):
    return model.makespan >= model.start_time[i, j] + sum(
        model.assignment[i, j, m] * processing_time[i, j, m] 
        for m in machines if (i, j, m) in processing_time
    )

model.makespan_constraint = Constraint(model.O, rule=makespan_rule)

# Objective Function
model.objective = Objective(expr=model.makespan, sense=minimize)

# Solve the model
solver = SolverFactory('gurobi')
solver.solve(model, tee=True)

# Display results
for job, operation in model.O:
    assigned_machine = next(m for m in machines if model.assignment[job, operation, m].value == 1)
    print(f"Job {job}, Operation {operation + 1} starts at {model.start_time[job, operation].value} on machine {assigned_machine}")
print(f"Makespan: {model.makespan.value}")

def plot_schedule(model, processing_time, machines):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.colormaps["tab10"]
    job_colors = {job: colors(job % 10) for job in model.J}
    legend_patches = []
    
    for (job, operation) in model.O:
        for m in machines:
            if (job, operation, m) in processing_time and model.assignment[job, operation, m].value > 0.5:
                start = model.start_time[job, operation].value
                duration = processing_time[job, operation, m]
                color = job_colors[job]
                
                ax.barh(m, duration, left=start, color=color, edgecolor='black', label=f'Job {job}' if f'Job {job}' not in legend_patches else None)
                ax.text(start + duration / 2, m, f'J{job}-O{operation+1}', va='center', ha='center', color='white', fontsize=10, fontweight='bold')
                
                if f'Job {job}' not in legend_patches:
                    legend_patches.append(f'Job {job}')
                break

    ax.set_xlabel("Time Units")
    ax.set_ylabel("Machines")
    ax.set_title("Job-Shop Schedule Gantt Chart")
    ax.set_yticks(machines)
    ax.grid(True, linestyle='--', alpha=0.7)

    handles = [plt.Line2D([0], [0], color=job_colors[job], lw=4, label=f'Job {job}') for job in model.J]
    ax.legend(handles=handles, title="Jobs", loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.show()

# Call the function after solving the model
plot_schedule(model, processing_time, machines)