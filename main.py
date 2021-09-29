from ap import *
from pgraph import *
from stats import *

import sys

import sys
from  ap_std import *

tests_min = [
    ("A1", lambda g: basic_hungarian(g, maximize=False)),
    ("B1", lambda g: multiple_hungarian(g, maximize=False)[0]),
    ("C1", lambda g: iterative_hungarian(g, maximize=False)[0]),
    ("D1", lambda g: greedy_hungarian(g, maximize=False)),
    ("E1", lambda g: iterative_hungarian_random_one(g, maximize=False)[0]),
    ("E9", lambda g: iterative_hungarian_random_one_best_of_n(g, 10, maximize=False)),
    ("F1", lambda g: iterative_hungarian_random_two(g, maximize=False)[0])


]


tests_min_quick = [
    ("A1", lambda g: basic_hungarian(g, maximize=False)),
    ("B1", lambda g: multiple_hungarian(g, maximize=False)[0]),
    ("C1", lambda g: iterative_hungarian(g, maximize=False)[0]),
    ("D1", lambda g: greedy_hungarian(g, maximize=False)),
    ("E1", lambda g: iterative_hungarian_random_one(g, maximize=False)[0]),
]


tests_min_slow = [
    ("E9", lambda g: iterative_hungarian_random_one_best_of_n(g, 10, maximize=False)),
    ("F1", lambda g: iterative_hungarian_random_two(g, maximize=False)[0])
]


tests_min_slowA = [
    ("A1", lambda g: basic_hungarian(g, maximize=False)),
    ("E9", lambda g: iterative_hungarian_random_one_best_of_n(g, 10, maximize=False)),
    ("F1", lambda g: iterative_hungarian_random_two(g, maximize=False)[0])
]

tests_max = [
    ("A1", lambda g: basic_hungarian(g, maximize=True)),
    ("B1", lambda g: multiple_hungarian(g, maximize=True)[0]),
    ("C1", lambda g: iterative_hungarian(g, maximize=True)[0]),
    ("D1", lambda g: greedy_hungarian(g, maximize=True)),
    ("E1", lambda g: iterative_hungarian_random_one(g, maximize=True)[0]),
    ("E9", lambda g: iterative_hungarian_random_one_best_of_n(g, 10, maximize=True)),
    ("F1", lambda g: iterative_hungarian_random_two(g, maximize=True)[0])
]



tests_max_quick = [
    ("A1", lambda g: basic_hungarian(g, maximize=True)),
    ("B1", lambda g: multiple_hungarian(g, maximize=True)[0]),
    ("C1", lambda g: iterative_hungarian(g, maximize=True)[0]),
    ("D1", lambda g: greedy_hungarian(g, maximize=True)),
    ("E1", lambda g: iterative_hungarian_random_one(g, maximize=True)[0]),
]



tests_max_slow = [
    ("E9", lambda g: iterative_hungarian_random_one_best_of_n(g, 10, maximize=True)),
    ("F1", lambda g: iterative_hungarian_random_two(g, maximize=True)[0])
]



tests_max_slowA = [
    ("A1", lambda g: basic_hungarian(g, maximize=True)),
    ("E9", lambda g: iterative_hungarian_random_one_best_of_n(g, 10, maximize=True)),
    ("F1", lambda g: iterative_hungarian_random_two(g, maximize=True)[0])
]


# numbr of tests, number of partitions, number of vertices, range of weights, min/max
random_data_params = {
    'random-max-3': (1000, 3, 30, (0,9),"max"),
    'random-min-3': (1000, 3, 30, (0,9), "min"),
    'random-max-4': (1000, 4, 100, (1,100), "max"),
    'random-min-4': (1000, 4, 100, (1,100), "min"),
}


random_data_params = {
  #  'random-min-3': (1000, 3, 30, (0,9), "min"),
    'random-min-4': (100, 4, 100, (1,100), "min"),
}
'''
A1 60.309 0.007014523506164551
B1 56.072 0.02110163354873657
C1 50.793 0.042954487085342406
D1 60.358 0.01690866708755493
E1 50.905 0.03750501799583435
E9 50.314 0.37163288140296935
F1 49.898 0.674991361618042
'''
#data_file_names = [
#    '3D1198N1','3DA198N3', '3DIJ99N2', '3D1198N2', '3DA99N1', '3DIJ99N3', '3D1198N3',
#    '3DA99N2', '3D1299N1', '3DA99N3', '3D1299N2', '3DI198N1', '3D1299N3', '3DI198N2','3DA198N1',
#    '3DI198N3', '3DA198N2',  '3DIJ99N1']


#danta_file_names = [
#'3DA198N3', '3DIJ99N2',	'3DA99N1', '3DIJ99N3',	'3DA99N2', '3DA99N3', '3DI198N1',
#'3DI198N2',	'3DA198N1',	'3DI198N3',	'3DA198N2',	'3DIJ99N1',
#]



problems0 = ['3DA198N1', '3DA198N2', '3DA198N3', '3DA99N1', '3DA99N2', '3DA99N3', '3DI198N1', '3DI198N2', '3DI198N3', '3DIJ99N1', '3DIJ99N2', '3DIJ99N3']
problems1 = ['3D1198N1', '3D1198N2', '3D1198N3', '3D1299N1', '3D1299N2', '3D1299N3']


def cost_time_mean(cts):
    L = 1.0 * len(cts)
    c, t = 0.0, 0.0
    for ct in cts:
        c += 1.0 * ct[0]
        t += 1.0 * ct[1]
    return c / L, t / L


#############
## GREEDY RANDOM
############
#for i in range(1000):
 #   g =  PGraph(4, 100, (1,100))
 #   M = [[[1]],[[2]],[[3]],[[4]],[[5]],[[6]]]
  #  print(AP3_greedy(M))


#exit()
######################################
# PLOTTING THE GRAPH
######################################


def wf(size):
    return 2.0**(np.random.randint(0,11,size=size))
    return (1.0*np.random.randint(0, size[0], size=size))

test_name = "F5-PLOT-EXP"

print("test", test_name)
# Get difference methods in percentage
#REP = 1000
#for P, REP in [(3,250),(4,100)]: # number of partitions
#for P, REP in [(4,50)]:
for P, REP in []:
  #  data0 = [["Vertices","Hungarian","sd","time","Multiple","sd","time","%","Iterative","sd","time","%"]]
    data0 = []
   # data0 = [["N","Multiple","Iteratieve"]]
    #for N in [2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,85,100]:# [2,3] + list(range(4,101,4)):
    for N in [60, 70, 85,  100]:  # [2,3] + list(range(4,101,4)):

    # if N <= 75: continue
            #range(2,(101 if P == 3 else 51)): # number of vertices per partition

        print("\nN =", N,end="")
        weight_range = (0,N-1)

       # times, costs = [[] for x in tests_max_quick], [[] for x in tests_max_quick]

        TESTS = tests_max_slowA

        times, costs = [[] for x in TESTS], [[] for x in TESTS]

        for i in range(REP): # repetitions
            print(".:"[i > REP // 2], end=["", "\n"][i % 200 == 199])
            sys.stdout.flush()


           # g = PGraph(P, N, weight_range)
            g = PGraph(P,N,wf=wf)
            # go through all tests
            for test_ind, (s, f) in enumerate(TESTS):
                start_timer(test_ind)
                costs[test_ind].append(f(g))
                times[test_ind].append(timer_value(test_ind))



        m1, s1, t1 = *fmeansd(costs[0]), fmean(times[0])
        m2, s2, t2 = *fmeansd(costs[1]), fmean(times[1])
        m3, s3, t3 = *fmeansd(costs[2]), fmean(times[2])
       # m4, s4, t4 = *fmeansd(costs[3]), fmean(times[3])
        #m5, s5, t5 = *fmeansd(costs[4]), fmean(times[4])

        p2 = 100.0*m2/m1
        p3 = 100.0*m3/m1
       # p4 = 100.0 * m4 / m1
       # p5 = 100.0*m5/m1

#   data0.append([N, m1, s1, t1, m2, s2, t2, p2, m3,s3, t3, p3])
        data0.append([N,1.0*m2/m1, 1.0*m3/m1])
#data0.append([N, 1.0 * m2 / m1, 1.0 * m3 / m1, 1.0 * m4 / m1, 1.0 * m5 / m1])

        csv_write(test_name+"-"+str(P)+"-"+str(REP)+"x.csv",data0)

#exit()



#######################################
######   TEST RANDOM INSTANCES  ALL ######
#######################################



for name in random_data_params:

    count, P, N, weight_range, mm = random_data_params[name]

  #  if P < 4:
   #     continue

    print("###",name,"###",P,N)
    tests = tests_max if mm == 'max' else tests_min_quick

    data0_dict = [[] for t in tests]



    # REPEAT (AVERAGE) LOOP
    for count_i in range(count):

        print(count_i, end=" ", flush=True)
        # generate graph
        g = PGraph(P, N, weight_range)

        # loop through the tests
        for test_ind, (method_name, f) in enumerate(tests):

            start_timer(test_ind)
            cost = f(g)

            data0_dict[test_ind].append((cost, timer_value(test_ind)))

            print(method_name[0], end="", flush=True)

        print("",end="\n", flush=True)


    # PRINT STATS

        print("\n STATS:\n")
        for test_ind, (method_name, f) in enumerate(tests):
            cts = data0_dict[test_ind]
            c, t = cost_time_mean(cts)
            print(method_name, c, t)

        print("")



print("\n *** FAST ONES ONES ***\n")

exit()

#######################################
######   TEST RANDOM INSTANCES - FAST ONES 4 parittions  ######
#######################################



for name in random_data_params:

    count, P, N, weight_range, mm = random_data_params[name]

    if P < 4:
        continue

    print("###",name,"###",P,N)
    tests = tests_max_quick if mm == 'max' else tests_min_quick

    data0_dict = [[] for t in tests]



    # REPEAT (AVERAGE) LOOP
    for count_i in range(count):

        print(count_i, end=" ", flush=True)
        # generate graph
        g = PGraph(P, N, weight_range)

        # loop through the tests
        for test_ind, (method_name, f) in enumerate(tests):

            start_timer(test_ind)
            cost = f(g)

            data0_dict[test_ind].append((cost, timer_value(test_ind)))

            print(method_name[0], end="", flush=True)

        print("",end="\n", flush=True)


    # PRINT STATS

    print("\n STATS:\n")
    for test_ind, (method_name, f) in enumerate(tests):
        cts = data0_dict[test_ind]
        c, t = cost_time_mean(cts)
        print(method_name, c, t)

    print("")





#######################################
######   TEST RANDOM INSTANCES slow ones  4 partitions  ######
#######################################

print("\n *** SLOW ONES ***\n")


for name in random_data_params:

    count, P, N, weight_range, mm = random_data_params[name]

    if P < 4:
        continue

    print("###",name,"###",P,N)
    tests = tests_max_slow if mm == 'max' else tests_min_slow

    data0_dict = [[] for t in tests]


    count //= 10


    # REPEAT (AVERAGE) LOOP
    for count_i in range(count):

        print(count_i, end=" ", flush=True)
        # generate graph
        g = PGraph(P, N, weight_range)

        # loop through the tests
        for test_ind, (method_name, f) in enumerate(tests):

            start_timer(test_ind)
            cost = f(g)

            data0_dict[test_ind].append((cost, timer_value(test_ind)))

            print(method_name[0], end="", flush=True)

        print("",end="\n", flush=True)


    # PRINT STATS

    print("\n STATS:\n")
    for test_ind, (method_name, f) in enumerate(tests):
        cts = data0_dict[test_ind]
        c, t = cost_time_mean(cts)
        print(method_name, c, t)

    print("")


#####################################
######   TEST KNOWN PROBLEMS   ######
#####################################

REPEAT = 100
#test_seq = tests_min_quick
test_seq = tests_min

for bin, names in []:#enumerate([problems0, problems1]):
    print("* BINARY *" if bin else "* NON-BINARY *")



    for problem_name in names:
        print(problem_name)



        P, N, g, best = load_pgraph(problem_name)

        cost_times = [[] for i in test_seq]

        for r in range(REPEAT):  # average for time and randomized tests

            print('\r'+str(r), end="", flush=True)



            for test_ind, (test_name, f) in enumerate(test_seq):
                start_timer(test_ind)
                cost = f(g)
                cost_times[test_ind].append((1.0*cost, 1.0*timer_value(test_ind)))

        print()
        for test_ind, (method_name, f) in enumerate(test_seq):
            c, t = cost_time_mean(cost_times[test_ind])
            print(method_name, c, t)







exit()

"""
for rep, P, N, weight_range in data_set_params:

    data0 = [["hungarian","time","multiple","time","iterative","time","iter2","time","iter3","time","iter4","time"]]

    # set up timer, etc
    times, costs = [[] for x in tests], [[] for x in tests]
    print("\nrepeat", rep, "P =", P, "N=", N, "data", str(weight_range))
    # loop through rep examples
    for i in range(rep):

        data0.append([])

        print(".:"[i>rep//2], end=["","\n"][i % 100 == 99])
        sys.stdout.flush()

        g = PGraph(P, N, weight_range)

        # go through all tests
        for test_ind, (s, f) in enumerate(tests):

            start_timer(test_ind)
            cost = f(g)

            data0[-1] += [cost, timer_value(test_ind)]
            times[test_ind].append(timer_value(test_ind))
            costs[test_ind].append(cost)

    csv_write("iteratives-P"+str(P)+"-N"+str(N)+"-w"+str(weight_range[0])+"-"+str(weight_range[1])+".csv",data0)

    print_stats([t[0] for t in tests], times, costs)

exit()


"""


# Loop through "known problems" data files


# TEST GREEDY
'''
for i, s in enumerate(data_file_names2):
    P, N, g, best = load_pgraph(s)
    print("Problem", i, "-", s, "P =", P, "N = ", N, "Weight range :", g.weight_range())
    cost = greedy_hungarian(g, maximize=False)
    print("D",cost)
    costa = basic_hungarian(g, maximize=False)
    print("A",costa)
    if costa < cost:
        print("JOJ!")
'''
#exit()



data0 = [["Problem name","Id","Partitions","N","Weight min","Weight max","Hungarian","t-H","Multiple Hungarian","t-M","Iterative hungarian","t-I","Random hungarian","t-I","Optimal solution"]]

cas = [[],[],[],[],[]]
for i, s in enumerate(data_file_names):

    for ijk in range(1):

        print(s)
        P, N, g, best = load_pgraph(s)

        data0.append([s,i,P,N,*g.weight_range()])
        print("Problem",i,"-",s, "P =",P, "N = ",N,"Weight range :",g.weight_range())
        #print("[", best,"]", end=' ')

        prev_test=  None

        COST = []
        TIME = []

        for test_ind, (ss, f) in enumerate(tests_min):
            start_timer(test_ind)
            cost = f(g)
            data0[-1].append(1.0*cost)
            data0[-1].append(1.0*timer_value(test_ind))
            cas[test_ind].append(timer_value(test_ind))
            print(s, "Cost",cost,"time",timer_value(test_ind))
            '''
            print(cost,end=" ")
                
            if prev_test is None:
                prev_test = cost
            else:
                if prev_test > cost:
                    print("        BETTER")
                elif prev_test == cost:
                    print("         SAME.")
                else:
                    print("       worst.")
    
            sys.stdout.flush()
            '''
            #cas[test_ind].append(timer_value(test_ind))

#    data0[-1].append(best)
 #   csv_write("known_graphs-vackrat.csv",data0) # write to csv
    #print()


print("final times")
print(sum(cas[0]))
print(sum(cas[1]))
print(sum(cas[2]))
print(sum(cas[3]))
print(sum(cas[4]))


exit(0)
"""

"""
# loop through data set parameters
for rep, P, N, weight_range in data_set_params:

    data0 = [["hungarian","time","multiple","time","iterative","time","iter2","time","iter3","time","iter4","time"]]

    # set up timer, etc
    times, costs = [[] for x in tests], [[] for x in tests]
    print("\nrepeat", rep, "P =", P, "N=", N, "data", str(weight_range))
    # loop through rep examples
    for i in range(rep):

        data0.append([])

        print(".:"[i>rep//2], end=["","\n"][i % 100 == 99])
        sys.stdout.flush()

        g = PGraph(P, N, weight_range)

        # go through all tests
        for test_ind, (s, f) in enumerate(tests):

            start_timer(test_ind)
            cost = f(g)

            data0[-1] += [cost, timer_value(test_ind)]
            times[test_ind].append(timer_value(test_ind))
            costs[test_ind].append(cost)

    csv_write("iteratives-P"+str(P)+"-N"+str(N)+"-w"+str(weight_range[0])+"-"+str(weight_range[1])+".csv",data0)

    print_stats([t[0] for t in tests], times, costs)

exit()
"""

# weight funtion
def wf(size):
  #  return 2.0**(np.random.randint(0,11,size=size))
    return (1.0*np.random.randint(0, size[0], size=size))


"""
#SHORT TEST FOR CHECKING
"""


for P, REP in [(3,500),(4,500)]: # number of partitions

    data0 = []
    for N in [2,3] + list(range(4,80,4)):

        print("Partitions =",P, "Repeats =",REP,"Vertices =",N,"Weights = (0,",N-1,")")

        weight_range = (0,N-1)

        '''
        ("basic hungarian    ", lambda g: basic_hungarian(g, maximize=False)),
        ("multiple hungarian ", lambda g: multiple_hungarian(g, maximize=False)[0]),
        ("iterative hungarian", lambda g: iterative_hungarian(g, maximize=False)[0]),
        ("iterative random", lambda g: iterative_hungarian_random(g, maximize=False)[0]),
        ("iterative random 1", lambda g: iterative_hungarian_random_one(g, maximize=False)[0]),
        ("iterative random 2", lambda g: iterative_hungarian_random_two(g, maximize=False)[0])
        '''

        costs_A1 = []
        costs_B1 = []
        costs_C1 = []

        for i in range(REP): # repetitions

            g = PGraph(P,N,wf=wf)
            # A1
            costs_B1.append(1.00 * multiple_hungarian(g, maximize=True)[0])
            costs_C1.append(1.00 * iterative_hungarian(g, maximize=True)[0])
            costs_A1.append(1.00 * basic_hungarian(g, maximize=True) )

        BdivA = 1.0*mean(costs_B1) / mean(costs_A1)
        CdivA = 1.0*mean(costs_C1) / mean(costs_A1)

        data0.append([N, BdivA, CdivA])

        csv_write("BIG2-MAX-INT-"+str(P)+".csv",data0)


exit()



test_name = "SHORT-EXP"

print("test")
# Get difference methods in percentage
#REP = 1000
for P, REP in [(3,1000),(4,1000)]: # number of partitions
#for P, REP in [(4,500)]:
  #  data0 = [["Vertices","Hungarian","sd","time","Multiple","sd","time","%","Iterative","sd","time","%"]]

    data0 = [["N","Multiple","Iteratieve"]]
    for N in [2,3] + list(range(4,101,4)):
       # if N <= 75: continue
            #range(2,(101 if P == 3 else 51)): # number of vertices per partition

        print("\nN =", N,end="")
        weight_range = (0,N-1)

        times, costs = [[] for x in tests], [[] for x in tests]

        for i in range(REP): # repetitions
            print(".:"[i > REP // 2], end=["", "\n"][i % 200 == 199])
            sys.stdout.flush()


           # g = PGraph(P, N, weight_range)
            g = PGraph(P,N,wf=wf)
            # go through all tests
            for test_ind, (s, f) in enumerate(tests):
                start_timer(test_ind)
                costs[test_ind].append(f(g))
                times[test_ind].append(timer_value(test_ind))



        m1, s1, t1 = *fmeansd(costs[0]), fmean(times[0])
        m2, s2, t2 = *fmeansd(costs[1]), fmean(times[1])
        m3, s3, t3 = *fmeansd(costs[2]), fmean(times[2])

        p2 = 100.0*m2/m1
        p3 = 100.0*m3/m1

     #   data0.append([N, m1, s1, t1, m2, s2, t2, p2, m3,s3, t3, p3])
        data0.append([N,1.0*m2/m1, 1.0*m3/m1])

        csv_write(test_name+"-"+str(P)+"-"+str(REP)+"x.csv",data0)

"""