
import itertools as it

# returns the cost of the cycle c, where costs are in matrices M
def cycle_cost(M, c):
    return sum( M[i][c[i]][c[(i+1)%len(M)]] for i in range(len(M)))



# Greedy algorithm: among all p-cycles select best ones that form a disjoint set
# input: list of matrices M = [costAB, costBC, cost CA]
# output: maximal cost and list of cycles
# TODO: generalize to p >= 2 parititons
def AP3_greedy(M):
    P = 3 # number of partitions
    N = len(M[0]) # number of vertices in each patition

    # list of all possible p-cycles, sorted by cost
    # elements of the list: (cost, cycle), where the cycle is a list of vertices
    p_cost = sorted([ (cycle_cost(M,c),c) for c in it.product(range(N), repeat = P)])

    disjoint_cycles = [] # list of cycles forming a maximal system
    used_vertices = [set() for i in range(P)] # used_vertices[i] = set of vetices already used in parition i
    total_cost = .0

    while len(disjoint_cycles) < N: # repeat until we have a full system
        cost, cycle = p_cost.pop() # get next most expensive cycle

        # are the cycle vertices disjoint from our current system?
        if all(v not in used_vertices[i] for i,v in enumerate(cycle)):
            # add new cycle to maximal system

            disjoint_cycles.append(cycle) # add cycle to system
            total_cost += cost # add to current cost
            for i, v in enumerate(cycle): # update the used vertices table
                used_vertices[i].add(v)

    return total_cost #, disjoint_cycles



# random solution
def AP3_random(M, tests = 1):

    sols = []
    for i in range(tests):
        N = len(M[0]) # number of vertices in each patition

        p,q,r = list(range(N)), list(range(N)),list(range(N))
        shuffle(q)
        shuffle(r)

        sols.append(M[0][p,q].sum() + M[1][q,r].sum() + M[2][r,p].sum())

    return max(sols)


def AP3_brute_force(M):
    P = 3  # number of partitions
    N = len(M[0])  # number of vertices in each patition
    system_costs = []
    for q2,q3 in it.product(it.permutations(range(N)), repeat=2):
        system_costs.append( sum( M[0][i][q2[i]]+ M[1][q2[i]][q3[i]] + M[2][q3[i]][i] for i in range(N)) )
    return max(system_costs)
