from datetime import datetime as dt
import time
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverManagerFactory
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import z3

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

jobs2 = {
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
        [(1, 5), (6, 6), (8, 4)],     				# Operation 6: Machines 1, 6, and 8, with times 5, 6, and 4
        [(4, 4)]                      				# Operation 7: Machine 4, with time 4
    ],                                				
    3: [                              				
        [(2, 7), (3, 6), (8, 8)],     				# Operation 1: Machines 2, 3, and 8, with times 7, 6, and 8
        [(4, 7), (8, 7)],             				# Operation 2: Machines 4 and 8, with times 7 and 7
        [(3, 7), (5, 8), (7, 7)],     				# Operation 3: Machines 3, 5, and 7, with times 7, 8, and 7
        [(4, 7), (6, 8)],             				# Operation 4: Machines 4 and 6, with times 7 and 8
        [(1, 1), (2, 4)]             				# Operation 5: Machines 1 and 2, with times 1 and 4  
    ],                                
    4: [                              
        [(1, 4), (3, 3), (5, 7)],     				# Operation 1: Machines 1, 3, and 5, with times 4, 3, and 7
        [(2, 4), (8, 4)],             				# Operation 2: Machines 2 and 8, with times 4 and 4
        [(3, 4), (4, 5), (6, 6), (7, 7)],  			# Operation 3: Machines 3, 4, 6, and 7, with times 4, 5, 6, and 7
        [(5, 3), (6, 5), (8, 5)]     				# Operation 4: Machines 5, 6, and 8, with times 3, 5, and 5
    ],                                
    5: [                              
        [(1, 3)],                     				# Operation 1: Machine 1, with time 3
        [(2, 4), (4, 5)],             				# Operation 2: Machines 2 and 4, with times 4 and 5
        [(3, 4), (8, 4)],             				# Operation 3: Machines 3 and 8, with times 4 and 4
        [(5, 3), (6, 3), (8, 3)],     				# Operation 4: Machines 5, 6, and 8, with times 3, 3, and 3
        [(4, 5), (6, 4)]             				# Operation 5: Machines 4 and 6, with times 5 and 4  				
    ],                                				
    6: [                              				
        [(1, 3), (2, 5), (3, 6)],     				# Operation 1: Machines 1, 2, and 3, with times 3, 5, and 6
        [(4, 7), (5, 8)],             				# Operation 2: Machines 4 and 5, with times 7 and 8
        [(3, 9), (6, 8)]             				# Operation 3: Machines 3 and 6, with times 9 and 8     
    ],                                
    7: [                              
        [(3, 4), (5, 5), (6, 4)],     				# Operation 1: Machines 3, 5, and 6, with times 4, 5, and 4
        [(4, 4), (7, 6), (8, 4)],     				# Operation 2: Machines 4, 7, and 8, with times 4, 6, and 4
        [(1, 3), (3, 3), (4, 4), (5, 5)],  			# Operation 3: Machines 1, 3, 4, and 5, with times 3, 3, 4, and 5
        [(4, 4), (6, 6), (8, 5)],     				# Operation 4: Machines 4, 6, and 8, with times 4, 6, and 5
        [(1, 3), (3, 3)]             				# Operation 5: Machines 1 and 3, with times 3 and 3
    ],                                
    8: [                              
        [(1, 3), (2, 4), (6, 4)],     				# Operation 1: Machines 1, 2, and 6, with times 3, 4, and 4
        [(4, 6), (5, 5), (8, 4)],     				# Operation 2: Machines 4, 5, and 8, with times 6, 5, and 4
        [(3, 4), (7, 5)],             				# Operation 3: Machines 3 and 7, with times 4 and 5
        [(4, 4), (6, 6)],             				# Operation 4: Machines 4 and 6, with times 4 and 6
        [(7, 1), (8, 2)]             				# Operation 5: Machines 7 and 8, with times 1 and 2
    ],
    9: [
        [(1, 5), (2, 6), (3, 4)],     				# Operation 1: Machines 1, 2, and 3, with times 5, 6, and 4
        [(4, 3), (5, 4), (6, 5)],     				# Operation 2: Machines 4, 5, and 6, with times 3, 4, and 5
        [(7, 6), (8, 5)],             				# Operation 3: Machines 7 and 8, with times 6 and 5
        [(1, 4), (3, 5), (5, 6)],     				# Operation 4: Machines 1, 3, and 5, with times 4, 5, and 6
        [(2, 3), (4, 4), (6, 5), (8, 4)]			# Operation 5: Machines 2, 4, 6, and 8, with times 3, 4, 5, and 4
    ],
    10: [
        [(2, 5), (4, 6), (6, 4), (8, 5)],			# Operation 1: Machines 2, 4, 6, and 8, with times 5, 6, 4, and 5
        [(1, 3), (3, 4), (5, 5), (7, 4)],			# Operation 2: Machines 1, 3, 5, and 7, with times 3, 4, 5, and 4
        [(2, 6), (4, 5), (6, 7)],     				# Operation 3: Machines 2, 4, and 6, with times 6, 5, and 7
        [(1, 4), (3, 5), (5, 6), (7, 5)],			# Operation 4: Machines 1, 3, 5, and 7, with times 4, 5, 6, and 5
        [(2, 5)],			                        # Operation 5: Machines 2 with time 5
        [(1, 6), (3, 5), (5, 7)]     				# Operation 6: Machines 1, 3, and 5, with times 6, 5, and 7
    ]
}

machines = [1, 2, 3, 4, 5, 6, 7, 8]

# Solvers to try -> IPOPT does not support integer constraints.
solvers = ['glpk', 'gurobi', 'ipopt', 'neos', 'z3']  # Add or remove solvers as needed

# Prompt the user to select a solver
print("Available Solvers:")
for i, solver in enumerate(solvers):
    print(f"{i}: {solver}")

while True:
    try:
        solver_num = int(input("Select a solver number: "))
        datasetX = int(input("Select Dataset 0 or 1: "))

        if datasetX == 1:
            jobs = jobs2

        if 0 <= solver_num < len(solvers):
            break
        else:
            print(f"Please enter a number between 0 and {len(solvers) - 1}.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

#Z3 Solver
def solve_job_shopZ3(jobs, machines):

    start_time = time.time()
    
    # Create a log of solver progress
    print("\n" + "="*50)
    print(f"Starting Z3 Solver at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    print("\nProblem Statistics:")
    print(f"Number of Jobs: {len(jobs)}")
    print(f"Number of Machines: {len(machines)}")
    total_ops = sum(len(ops) for ops in jobs.values())
    print(f"Total Operations: {total_ops}")
    print("-"*50 + "\n")

    solver = z3.Optimize()
    
    z3.set_param('verbose', 10)
    z3.trace_file = 'z3_trace.log'
    print("Creating variables and constraints...")
    
    # Create variables
    # start_time[job][operation] represents the start time of each operation
    start_time = {(j, o): z3.Int(f'start_{j}_{o}') 
                 for j in jobs.keys() 
                 for o in range(len(jobs[j]))}
    
    # assignment[job][operation][machine] represents whether operation is assigned to machine
    assignment = {(j, o, m): z3.Bool(f'assign_{j}_{o}_{m}')
                 for j in jobs.keys()
                 for o in range(len(jobs[j]))
                 for m, _ in jobs[j][o]}
    
    # Define makespan variable
    makespan = z3.Int('makespan')
    
    # Add constraints
    
    # 1. All start times must be non-negative
    for start in start_time.values():
        solver.add(start >= 0)
    
    # 2. Each operation must be assigned to exactly one machine
    for j in jobs:
        for o in range(len(jobs[j])):
            # Sum of assignments for this operation must be 1
            solver.add(z3.Sum([z3.If(assignment[j, o, m], 1, 0) 
                             for m, _ in jobs[j][o]]) == 1)
    
    # 3. Precedence constraints between operations of the same job
    for j in jobs:
        for o in range(1, len(jobs[j])):
            prev_o = o - 1
            # Previous operation's completion time
            prev_completion = start_time[j, prev_o] + z3.Sum([
                z3.If(assignment[j, prev_o, m], t, 0)
                for m, t in jobs[j][prev_o]
            ])
            # Current operation must start after previous operation ends
            solver.add(start_time[j, o] >= prev_completion)
    
    # 4. No overlap constraints between operations on the same machine
    for m in machines:
        # Get all operations that can be assigned to this machine
        machine_ops = [(j, o) for j in jobs 
                      for o in range(len(jobs[j])) 
                      for m_possible, _ in jobs[j][o] 
                      if m_possible == m]
        
        for i, (j1, o1) in enumerate(machine_ops):
            for j2, o2 in machine_ops[i+1:]:
                # Get processing times for these operations on this machine
                t1 = next(t for m_pos, t in jobs[j1][o1] if m_pos == m)
                t2 = next(t for m_pos, t in jobs[j2][o2] if m_pos == m)
                
                # If both operations are assigned to this machine,
                # they cannot overlap
                solver.add(z3.Or(
                    z3.Not(assignment[j1, o1, m]),
                    z3.Not(assignment[j2, o2, m]),
                    start_time[j1, o1] + t1 <= start_time[j2, o2],
                    start_time[j2, o2] + t2 <= start_time[j1, o1]
                ))
    
    # 5. Makespan constraints
    for j in jobs:
        for o in range(len(jobs[j])):
            # Completion time of each operation
            completion_time = start_time[j, o] + z3.Sum([
                z3.If(assignment[j, o, m], t, 0)
                for m, t in jobs[j][o]
            ])
            solver.add(makespan >= completion_time)
    
    # Objective: Minimize makespan
    solver.minimize(makespan)
    
    # Solve the problem
    if solver.check() == z3.sat:
        model = solver.model()
        
        # Extract solution
        solution = {
            'makespan': model[makespan].as_long(),
            'schedule': []
        }
        
        for j in jobs:
            for o in range(len(jobs[j])):
                start = model[start_time[j, o]].as_long()
                assigned_machine = None
                duration = None
                
                for m, t in jobs[j][o]:
                    if model[assignment[j, o, m]]:
                        assigned_machine = m
                        duration = t
                        break
                
                solution['schedule'].append({
                    'job': j,
                    'operation': o + 1,
                    'machine': assigned_machine,
                    'start': start,
                    'duration': duration,
                    'end': start + duration
                })
        
        return solution
    else:
        return None

#Z3 Print solution
def print_schedule_resultsZ3(solution):
    """Print detailed schedule results in a formatted way"""
    if not solution:
        print("No solution found.")
        return
    
    print("\n" + "="*50)
    print("SCHEDULE RESULTS")
    print("="*50)
    print(f"Optimal Makespan: {solution['makespan']} time units")
    print("-"*50)
    
    # Sort schedule by start time
    schedule = sorted(solution['schedule'], key=lambda x: x['start'])
    
    # Print schedule in a tabulated format
    print("\nDetailed Schedule:")
    print(f"{'Job':^5} {'Op':^5} {'Machine':^8} {'Start':^10} {'End':^10} {'Duration':^10}")
    print("-"*50)
    for item in schedule:
        print(f"{item['job']:^5} {item['operation']:^5} {item['machine']:^8} "
              f"{item['start']:^10} {item['end']:^10} {item['duration']:^10}")
    
    # Print machine utilization
    print("\nMachine Utilization:")
    print("-"*50)
    for m in machines:
        machine_ops = [s for s in schedule if s['machine'] == m]
        total_time = sum(op['duration'] for op in machine_ops)
        utilization = (total_time / solution['makespan']) * 100 if solution['makespan'] > 0 else 0
        print(f"Machine {m}: {utilization:.1f}% utilized")

#Z3 Print Plot
def plot_scheduleZ3(solution, machines):
    if not solution:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.get_cmap("tab10")
    job_colors = {}
    patches = {}
    
    for item in solution['schedule']:
        j = item['job']
        if j not in job_colors:
            job_colors[j] = colors(j % 10)
        color = job_colors[j]
        
        # Draw the operation bar
        ax.barh(item['machine'], item['duration'], 
                left=item['start'], color=color, edgecolor='black')
        
        # Add operation label
        ax.text(item['start'] + item['duration']/2, item['machine'], 
                f'O{item["operation"]}', va='center', ha='center',
                color='white', fontsize=10, fontweight='bold')
        
        if j not in patches:
            patches[j] = mpatches.Patch(color=color, label=f'Job {j}')
    
    ax.set_xlabel("Time Units")
    ax.set_ylabel("Machines")
    ax.set_title("Job-Shop Schedule Gantt Chart")
    ax.set_yticks(machines)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    handles = [patches[j] for j in sorted(patches.keys())]
    ax.legend(handles=handles,
             title="Jobs",
             loc='upper right',
             fontsize='medium',
             handlelength=1.5,
             handletextpad=0.5,
             bbox_to_anchor=(1.15, 1),
             borderaxespad=0.)
    
    plt.tight_layout()
    return fig

#Others
def solve_job_shop(jobs, machines, solver_num):
    
    # Model definition
    model = ConcreteModel()

    # Sets
    model.J = Set(initialize=jobs.keys(), doc="Jobs")
    model.M = Set(initialize=machines, doc="Machines")
    model.O = Set(initialize=[(job, operation) for job in jobs for operation in range(len(jobs[job]))], doc="Operations")

    # Parameters
    processing_time = {(job, op, m): t for job in jobs for op, times in enumerate(jobs[job]) for m, t in times}

    # Decision Variables
    model.start_time = Var(model.O, within=NonNegativeIntegers, doc="Start time of each operation")
    model.assignment = Var(model.O, model.M, within=Binary, doc="Assignment of operations to machines")
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

    # Prevent overlap using big-M method for constraint activation

    # Precedence_no_overlap_rule ensures that if a precedence exists, the earlier operation must finish before the later one starts.
    def precedence_no_overlap_rule(model, j1, o1, j2, o2, m):
        if (j1, o1) < (j2, o2) and j1 != j2 and (j1, o1, m) in processing_time and (j2, o2, m) in processing_time:

            # Removes the constraint when precedence, assignment1, or assignment2 are not satisfied.
            # j1 o1 must finish before j2 o2 starts + processing time of j2 - large negative adjustment that effectively deactivates the constraint under certain conditions
            return model.start_time[j2, o2] >= model.start_time[j1, o1] + processing_time[j1, o1, m] - (3 - model.precedence[j1, o1, j2, o2] - model.assignment[j1, o1, m] - model.assignment[j2, o2, m] ) * M
        return Constraint.Skip

    # Precedence_no_overlap_no_precedence_rule ensures that if there is no precedence, an operation cannot start before another one finishes.
    def precedence_no_overlap_no_precedence_rule(model, j1, o1, j2, o2, m):
        if (j1, o1) < (j2, o2) and j1 != j2 and (j1, o1, m) in processing_time and (j2, o2, m) in processing_time:

            # This ensures that if precedence = 0, the constraint forces (j1, o1) to start after (j2, o2) finishes.
            return model.start_time[j1, o1] >= model.start_time[j2, o2] + processing_time[j2, o2, m] - (2 + model.precedence[j1, o1, j2, o2] - model.assignment[j1, o1, m] - model.assignment[j2, o2, m] ) * M
        return Constraint.Skip

    # 3- Add constraint no Overlap with Precedence and without Precedence
    model.Precedence_No_Overlap = Constraint(model.O, model.O, model.M, rule=precedence_no_overlap_rule)
    model.Precedence_No_OverlapWithoutPrecedences = Constraint(model.O, model.O, model.M, rule=precedence_no_overlap_no_precedence_rule)

    # 4. Makespan definition
    # This constraint ensures that the makespan is always greater than or equal to every operation's start time + processing time
    def makespan_rule(model, j, o):
        return model.makespan >= model.start_time[j, o] + sum(
            model.assignment[j, o, m] * processing_time[j, o, m] for m in model.M if (j, o, m) in processing_time
        )
    model.Makespan_Constraint = Constraint(model.O, rule=makespan_rule)

    # Objective Function
    model.objective = Objective(expr=model.makespan, sense=minimize)

    return model, processing_time

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


    # Solve Others
    solver_name = solvers[solver_num]  # Assign selected solver

    # Objective Function
    model.objective = Objective(expr=model.makespan, sense=minimize)

    try:
        if solver_num != 3 and solver_num < 4:
            solver = SolverFactory(solver_name)
            if solver is None:  # Check if the solver is available
                print(f"Solver '{solver_name}' not found. Skipping.")

        if solver_num == 0:  # GLPK
            results = solver.solve(model, tee=True, options={
                'mipgap': 0.01,
                'tmlim': 300  # 5-minute time limit,            
                }) 
            # mipgap option is essential for controlling the trade-off between solution quality and solution time. It allows you to get good, near-optimal solutions within a reasonable time frame.
        elif solver_num == 3:  # NEOS
            solver_manager = SolverManagerFactory('neos')  # Creates a NEOS solver manager
            # print(solver_manager.available())  # This should list available solvers
            results = solver_manager.solve(model, solver='cplex', tee=True)
            #model.display()
        else: 
            results = solver.solve(model, tee=True)  # tee=True for solver output
        
        if results.solver.status == SolverStatus.ok:
            print(f"Solution found using {solver_name}!")
            print_schedule_results(model, jobs)  # Your existing function
            
            if solver_num != 2:
                fig = plot_schedule(model, processing_time, machines, jobs)
                if fig:
                    plt.show()
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            print(f"Model is infeasible with {solver_name}.")
        elif results.solver.termination_condition == TerminationCondition.unbounded:
            print(f"Model is unbounded with {solver_name}.")
        else:
            print(f"Solver {solver_name} did not find an optimal solution. Status: {results.solver.termination_condition}")

    except Exception as e:
        print(f"An error occurred with solver {solver_name}: {e}")

if solver_num == 4:  # Z3
    # Solve and visualize
    solution = solve_job_shopZ3(jobs, machines)
    print_schedule_resultsZ3(solution)
    fig = plot_scheduleZ3(solution, machines)
    if fig:
        plt.show()
else:
    model, processing_time = solve_job_shop(jobs, machines, solver_num)

     # Solve Others
    solver_name = solvers[solver_num]  # Assign selected solver

    try:
        if solver_num != 3 and solver_num < 4:
            solver = SolverFactory(solver_name)
            if solver is None:  # Check if the solver is available
                print(f"Solver '{solver_name}' not found. Skipping.")

        if solver_num == 0:  # GLPK
            results = solver.solve(model, tee=True, options={
                'mipgap': 0.01,
                'tmlim': 300  # 5-minute time limit,            
                }) 
            # mipgap option is essential for controlling the trade-off between solution quality and solution time. It allows you to get good, near-optimal solutions within a reasonable time frame.
        elif solver_num == 3:  # NEOS
            solver_manager = SolverManagerFactory('neos')  # Creates a NEOS solver manager
            # print(solver_manager.available())  # This should list available solvers
            results = solver_manager.solve(model, solver='cplex', tee=True)
            #model.display()
        else: 
            results = solver.solve(model, tee=True)  # tee=True for solver output
        
        if results.solver.status == SolverStatus.ok:
            print(f"Solution found using {solver_name}!")
            print_schedule_results(model, jobs)  # Your existing function
            
            if solver_num != 2:
                fig = plot_schedule(model, processing_time, machines, jobs)
                if fig:
                    plt.show()
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            print(f"Model is infeasible with {solver_name}.")
        elif results.solver.termination_condition == TerminationCondition.unbounded:
            print(f"Model is unbounded with {solver_name}.")
        else:
            print(f"Solver {solver_name} did not find an optimal solution. Status: {results.solver.termination_condition}")

    except Exception as e:
        print(f"An error occurred with solver {solver_name}: {e}")

