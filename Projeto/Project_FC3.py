from pyomo.environ import *

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

machines = [1, 2, 3, 4, 5, 6, 7, 8]  # Available machines

# Define the model
model = ConcreteModel()

# Sets
model.Jobs = Set(initialize=jobs.keys())  # Set of jobs
model.Operations = Set(initialize=lambda model: [(j, o + 1) for j in jobs for o in range(len(jobs[j]))])  # (Job, Operation)
model.Machines = Set(initialize=machines)  # Set of machines

# Parameters
def operation_data_init(model, j, o, m):
    for machine, time in jobs[j][o - 1]:  # o-1 because operation index starts at 1
        if machine == m:
            return time
    return 0

model.processing_time = Param(model.Operations, model.Machines, initialize=operation_data_init, within=NonNegativeReals, default=0)

# Variables
model.start_time = Var(model.Operations, within=NonNegativeReals)
model.makespan = Var(within=NonNegativeReals)

# Objective: Minimize the makespan
model.obj = Objective(expr=model.makespan, sense=minimize)

# Constraints
def precedence_constraint_rule(model, j, o):
    """Ensure that operations within the same job are processed in sequence."""
    if o < len(jobs[j]):
        return model.start_time[j, o + 1] >= model.start_time[j, o] + sum(
            model.processing_time[j, o, m] for m in model.Machines
        )
    return Constraint.Skip

model.precedence_constraint = Constraint(model.Operations, rule=precedence_constraint_rule)

def machine_availability_constraint_rule(model, m):
    """Ensure that a machine can process one operation at a time."""
    return sum(
        model.processing_time[j, o, m] * (model.start_time[j, o] + model.processing_time[j, o, m] <= model.makespan)
        for (j, o) in model.Operations
    ) <= 1

model.machine_availability_constraint = Constraint(model.Machines, rule=machine_availability_constraint_rule)

def makespan_constraint_rule(model, j, o):
    """Ensure the makespan is the maximum completion time of all operations."""
    return model.makespan >= model.start_time[j, o] + sum(
        model.processing_time[j, o, m] for m in model.Machines
    )

model.makespan_constraint = Constraint(model.Operations, rule=makespan_constraint_rule)

# Solve the model
solver = SolverFactory('glpk')
results = solver.solve(model)

# Print the results
print("Makespan:", model.makespan())
for j, o in model.Operations:
    print(f"Job {j} - Operation {o} starts at time {model.start_time[j, o].value}")