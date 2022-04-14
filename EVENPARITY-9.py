# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:55:15 2020

@author: Aliyu Sambo
"""


# -*- coding: utf-8 -*-
"""

This problem is a classification example using STGP (Strongly Typed Genetic Pro-
gramming). The evolved programs work on floating-point values AND Booleans values.
 The programs must return a Boolean value which must be true if e-mail is spam,
 and false otherwise. It uses a base of emails (saved in spambase.csv, see 
 Reference), from which it randomly chooses 400 emails to evaluate each individual.
 
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, trgY), float, "x") #EnCool
trgY = len(Tpoints[1])-1

"""
# Initialise --------------------------
import copy
import random
import numpy
from functools import partial
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import csv
import itertools
import operator
#from . import tools
import pandas as pd
import numpy as np
from functools import reduce
from operator import add, itemgetter
from multiprocessing.pool import ThreadPool, threading
import os
import datetime
import time

run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") #           (2)

# PARAMETERS -----------------------------------
runs = 50
# runs = 2

popsize=200
cxpb, mutpb, ngen = 0.7, 0.3, 50
# data = 'diabetes1'
# Prob = 'Progm'

Prob = 'EVENPARITY'
trgt=0.67 # 9

###############################################################################
#--------------------------------------------------
devicename = os.environ['COMPUTERNAME']
if devicename == 'DESKTOP-MLNSBQ2':
    system = 'laptop'
elif devicename == 'DESKTOP-JAN9GCB':
    system =    'NBOOK'
elif devicename == 'DESKTOP-4VA0QI6':
    system = 'Dktop2'  
else: system = 'desktop'
#--------------------------------------------------
if system == 'server':
    result_dir = f'/home/aliyu/Documents/APGP_FL/'
elif system == 'desktop':	
    result_dir = f'C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Ph6\\Test\\'
elif system == 'laptop':
    result_dir = f'C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Ph6\\Test\\'
elif system == 'NBOOK':
    result_dir = f'C:\\Users\\Aliyu Sambo\\OneDrive - Birmingham City University\\Experiment_Ph6\\Test\\'
elif system == 'Dktop2':
    result_dir = f"C:\\Users\\user\\Documents\\Ph6\\prog_result\\"
#--------------------------------------------------
###############################################################################


# Initialize Parity problem input and output matrices
PARITY_FANIN_M = 9
PARITY_SIZE_M = 2**PARITY_FANIN_M

inputs = [None] * PARITY_SIZE_M
outputs = [None] * PARITY_SIZE_M

for i in range(PARITY_SIZE_M):
    inputs[i] = [None] * PARITY_FANIN_M
    value = i
    dividor = PARITY_SIZE_M
    parity = 1
    for j in range(PARITY_FANIN_M):
        dividor /= 2
        if value >= dividor:
            inputs[i][j] = 1
            parity = int(not parity)
            value -= dividor
        else:
            inputs[i][j] = 0
    outputs[i] = parity

pset = gp.PrimitiveSet("MAIN", PARITY_FANIN_M, "IN")
pset.addPrimitive(operator.and_, 2)
pset.addPrimitive(operator.or_, 2)
pset.addPrimitive(operator.xor, 2)
pset.addPrimitive(operator.not_, 1)
pset.addTerminal(1)
pset.addTerminal(0)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=3, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalParity(individual):
    time_st = time.perf_counter() # <-- data only - start reading after compilation
    
    try:
        func = toolbox.compile(expr=individual)
        result = sum(func(*in_) == out for in_, out in zip(inputs, outputs))/PARITY_SIZE_M
    except (MemoryError, ZeroDivisionError, ValueError, TypeError):#    except ZeroDivisionError:
        result = 0.010101010101010101
        #print('MemoryError during evalution')
    evln_sec=float((time.perf_counter() - time_st))
    
    return result, evln_sec,


toolbox.register("evaluate", evalParity)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select2", tools.selDoubleTournament)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

############################################################################### 



"""    
#==============================================================================
#==============================================================================
"""
#--------------------------------------------------
#       STEADY-STATE
#--------------------------------------------------

"""
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
# Standard GP - Steady State
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
"""
def gpSteadyState(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, datatrain=None, datatest=None, target =None):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
    mettarget = 0 # Marker to indicate when target has been met; 0 = not set

    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    
    logbook = tools.Logbook()
    logbook.header = ['run', 'gen', 'nevals'] + (stats.fields if stats else [])
#+++++++++++++++++++++++++++++++++++++++++++++
#Evaluation of Initial Population
#+++++++++++++++++++++++++++++++++++++++++++++
    
    # Evaluate the individuals with an invalid fitness     
    counteval = 0
    for ind in population:
        if not ind.fitness.valid:
            counteval =+ 1
            # xo, yo, zo, = toolbox.evaluate(ind, datatrain, datatest)
            xo, yo, = toolbox.evaluate(ind)#, datatrain, datatest)
            ind.fitness.values = xo,          
            ind.evlntime = yo,
            # ind.testfitness = zo,

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    # logbook.record(gen=0, nevals=len(invalid_ind), **record)
    logbook.record(gen=0, nevals=counteval, **record)
    if verbose:
        print(logbook.stream)

    gen=0
    hof_db=[]
    hof_db.append([run, gen, halloffame[0].fitness.values[0], #halloffame[0].testfitness[0],
                   halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0])])
  
    # hof_db.append([run, gen, halloffame[0].fitness.values[0], halloffame[0].testfitness[0],
    #                halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0])])
  
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
    from operator import attrgetter

    def selInverseTournament(individuals, k, tournsize, fit_attr="fitness"):
        """Select the worst individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for i in range(k):
            aspirants =  [random.choice(individuals) for i in range(tournsize)]
            chosen.append(min(aspirants, key=attrgetter(fit_attr)))
#            for l in aspirants: print(str(l.fitness))
        return chosen
#+++++++++++++++++++++++++++++++++++++++++++++
#Breeding Function
#+++++++++++++++++++++++++++++++++++++++++++++
# define a breed function as nested.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, mettarget

    # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
    # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
        """
        Modified to address (the error is occassional e.g. 1 in 15 runs):
		MemoryError: DEAP : Error in tree evaluation : Python cannot evaluate a tree higher than 90. 
        If error encountered try breeding again. 
		"""
        successful = False
        while successful == False:
            try: 	
                
                #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
                p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))
        
                #++++++++ Crossover +++++++++++++++++++++++++++++++++++++
                if random.random() < cxpb:
                    p1, p2 = toolbox.mate(p1, p2)
                    del p1.fitness.values
        
                #++++++++ mutation on the offspring ++++++++++++++++               
                if random.random() < mutpb:
                    p1, = toolbox.mutate(p1)
                    del p1.fitness.values
        
                # Evaluate the offspring if it has changed
                if not p1.fitness.valid:
                    #++++++++ Counting evaluations +++++++++++++++++
                    # counteval_lock.acquire()
                    # counteval += 1 #Count the actual evaluations
                    # counteval_lock.release()
                    # xo, yo, zo = toolbox.evaluate(p1, datatrain, datatest)
                    xo, yo = toolbox.evaluate(p1)
        
                    # xo, yo = toolbox.evaluate(p1, datatrain, datatest)
                    p1.evlntime = yo,
                    # p1.testfitness = zo,
                    p1.fitness.values = xo, 
                    #Check if ZeroDivisionError, ValueError 
                    if p1.fitness.values == (0.0101010101010101,) :
                        p1.fitness.values = 0.0, #for maximising
                    # if p1.testfitness == (0.0101010101010101,) :
                    #     p1.testfitness = 0.0, #for maximising  
        #                print('check this out')
        #                print(str(p1))
        #                print(str(p1.fitness.values))

                    #++++++++ Counting evaluations +++++++++++++++++
                    counteval_lock.acquire()
                    counteval += 1 #Count the actual evaluations
                    counteval_lock.release()
                    successful = True
					
            except MemoryError :
                print('MemoryError encountered')
                successful = False
    # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
    # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
                
#[[[[[[[[[[[[[[[[[[[[[[[[[ non APGP [[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
#[[[[[[[[[[[[[[[[[[[[[[[[[ non APGP [[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        # if float(p1.fitness.values[0]) >= target:
        if float(p1.fitness.values[0]) >= target:
            #                print('Hi')
            if mettarget == 0:
                prev_gen = (len(population))*(gen-1)
                mettarget = counteval + prev_gen
                # print(f'Current Gen: {gen}, EVLN={counteval}')
                mettarget = counteval + prev_gen
                print(f'Target met: {mettarget}')
                print(f'Fitness: {float(p1.fitness.values[0])}')
                targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.fitness.values[0]), 'Met_at': mettarget}, index = {run})
            
                target_csv = f'{report_csv[:-4]}_Target.csv'
                #Export from dataframe to CSV file. Update if exists
                if os.path.isfile(target_csv):
                    targetmet_df.to_csv(target_csv, mode='a', header=False)
                else:
                    targetmet_df.to_csv(target_csv)                    
#MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
#[[[[[[[[[[[[[[[[[[[[[[[[[ non APGP [[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]  
#[[[[[[[[[[[[[[[[[[[[[[[[[ non APGP [[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END
                    
        #+++++++++++++++++++++++++++++++++++++++++++++
#       Identify an individual to be replaced - worst fitness
        #+++++++++++++++++++++++++++++++++++++++++++++
#            p1, p2 = list(map(toolbox.clone, random.sample(population, 2)))
        #+++++++++++++++++++++++++++++++++++++++++++++
#        update_lock.acquire()          # LOCK !!!  
        # Identify a individual to replace from the population. Use Inverse Tournament
        candidates = selInverseTournament(population, k=1, tournsize=5)
        candidate = candidates[0]
        # Replace if offspring is better than candidate individual 
        if p1.fitness.values[0] > candidate.fitness.values[0]: # Max
        # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                population.append(p1) 
                population.remove(candidate)
        
#        update_lock.release()            # RELEASE !!!
        #+++++++++++++++++++++++++++++++++++++++++++++

#    Update hall of fame   ????==== INDENT TO HAPPEN ONLY IF A RELPLACEMENT IS DONE ====?????                                                                     
        try:
            halloffame.update(population)
        except AttributeError:
            pass  

###############################################################################
    
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Collect Stats for the Generation 
    #+++++++++++++++++++++++++++++++++++++++++++++      
    def collectStatsGen():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
        #++++++++++ Collect Stats ++++++++++++++++++++
        record = stats.compile(population) if stats else {}
        logbook.record(run= run, gen=gen, nevals=counteval, **record)
        
        if verbose:
            print(logbook.stream) 
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Update hall of fame database for each generation
        hof_db.append([run, gen, halloffame[0].fitness.values[0], #halloffame[0].testfitness[0], 
                       halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0])])
        # hof_db.append([run, gen, halloffame[0].fitness.values[0], halloffame[0].testfitness[0], 
        #                halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0])])
                
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Create Report for the Run 
    #+++++++++++++++++++++++++++++++++++++++++++++        
    def collectStatsRun():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 

        #Put into dataframe
        chapter_keys = logbook.chapters.keys()
        sub_chaper_keys = [c[0].keys() for c in logbook.chapters.values()]
        
        data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                     in zip(sub_chaper_keys, logbook.chapters.values())]
        # data = np.array([[*a, *b, *c, *d] for a, b, c, d in zip(*data)])
        data = np.array([[*a, *b, *c] for a, b, c in zip(*data)])

        columns = reduce(add, [["_".join([x, y]) for y in s] 
                               for x, s in zip(chapter_keys, sub_chaper_keys)])
        df = pd.DataFrame(data, columns=columns)
        
        keys = logbook[0].keys()
        data = [[d[k] for d in logbook] for k in keys]
        for d, k in zip(data, keys):
            df[k] = d
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Export Report to local file
        if os.path.isfile(report_csv):
            df.to_csv(report_csv, mode='a', header=False)
        else:
            df.to_csv(report_csv)
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        ## Save 'Hall Of Fame' database
        #++++++++++++++++++++++++++++++++++++++++++++++
        #List to dataframe
        # hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
        hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Evln_time', 'Length', 'Best'])
        #Destination file (local)
        hof_csv = f'{report_csv[:-4]}_hof.csv'
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(hof_csv):
            hof_dframe.to_csv(hof_csv, mode='a', header=False)
        else:
            hof_dframe.to_csv(hof_csv)
###############################################################################


#+++++++++++++++++++++++++++++++++++++++++++++
#Create a Generation
#+++++++++++++++++++++++++++++++++++++++++++++
    # Begin the generational process
    for gen in range(1, ngen+1):
        # Generate offsprings -  equivalent to a generation / populations size
        poplnsize =  len(population)
#        poplnsize =  500
        counteval = 0 
        for h in range(poplnsize):
            breed()
        
        while counteval < poplnsize:
#            print(f'more pending')
            for j in range(poplnsize - counteval):
                breed()

        collectStatsGen()
    collectStatsRun()
    
###############################################################################       
    return population, logbook    
###############################################################################



"""
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
# Double Tournament GP - Steady State
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
"""
def gpDoubleT(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, datatrain=None, datatest=None, target =None):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
    mettarget = 0 # Marker to indicate when target has been met; 0 = not set
    
    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    
    logbook = tools.Logbook()
    logbook.header = ['run', 'gen', 'nevals'] + (stats.fields if stats else [])

#+++++++++++++++++++++++++++++++++++++++++++++
#Evaluation of Initial Population
#+++++++++++++++++++++++++++++++++++++++++++++
    
    # Evaluate the individuals with an invalid fitness     
    counteval = 0
    for ind in population:
        if not ind.fitness.valid:
            counteval =+ 1
            # xo, yo, zo, = toolbox.evaluate(ind, datatrain, datatest)
            xo, yo,  = toolbox.evaluate(ind)
            ind.fitness.values = xo,          
            ind.evlntime = yo,
            # ind.testfitness = zo,

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    # logbook.record(gen=0, nevals=len(invalid_ind), **record)
    logbook.record(gen=0, nevals=counteval, **record)
    if verbose:
        print(logbook.stream)

    gen=0
    hof_db=[]
    hof_db.append([run, gen, halloffame[0].fitness.values[0], #halloffame[0].testfitness[0],
                   halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0])])#
    # hof_db.append([run, gen, halloffame[0].fitness.values[0], halloffame[0].testfitness[0],
    #                halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0])])#

#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
    from operator import attrgetter

    def selInverseTournament(individuals, k, tournsize, fit_attr="fitness"):
        """Select the worst individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for i in range(k):
            aspirants =  [random.choice(individuals) for i in range(tournsize)]
            chosen.append(min(aspirants, key=attrgetter(fit_attr)))
#            for l in aspirants: print(str(l.fitness))
        return chosen
#+++++++++++++++++++++++++++++++++++++++++++++
#Breeding Function
#+++++++++++++++++++++++++++++++++++++++++++++
# define a breed function as nested.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, mettarget

    # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
    # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
        """
        Modified to address (the error is occassional e.g. 1 in 15 runs):
		MemoryError: DEAP : Error in tree evaluation : Python cannot evaluate a tree higher than 90. 
        If error encountered try breeding again. 
		"""
        successful = False
        while successful == False:
            try: 	
#            print(f'more pending')
                #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
                p1, p2 = list(map(toolbox.clone, toolbox.select2(population, 2, fitness_size=3, parsimony_size=1.4, fitness_first=False)))
    
                #++++++++ Crossover +++++++++++++++++++++++++++++++++++++
                if random.random() < cxpb:
                    p1, p2 = toolbox.mate(p1, p2)
                    del p1.fitness.values
    
                #++++++++ mutation on the offspring ++++++++++++++++               
                if random.random() < mutpb:
                    p1, = toolbox.mutate(p1)
                    del p1.fitness.values

                # Evaluate the offspring if it has changed
                if not p1.fitness.valid:

                    # xo, yo, zo = toolbox.evaluate(p1, datatrain, datatest)
                    xo, yo = toolbox.evaluate(p1)
                    p1.evlntime = yo,
                    # p1.testfitness = zo,
                    p1.fitness.values = xo, 
                    #Check if ZeroDivisionError, ValueError 
                    if p1.fitness.values == (0.0101010101010101,) :
                        p1.fitness.values = 0.0, #for maximising
                    # if p1.testfitness == (0.0101010101010101,) :
                    #     p1.testfitness = 0.0, #for maximising  
        #                print('check this out')
        #                print(str(p1))
        #                print(str(p1.fitness.values))
                    #++++++++ Counting evaluations +++++++++++++++++
                    counteval_lock.acquire()
                    counteval += 1 #Count the actual evaluations
                    counteval_lock.release()
                    successful = True
					
            except MemoryError :
                print('MemoryError encountered')
                successful = False
    # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
    # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
                
#[[[[[[[[[[[[[[[[[[[[[[[[[ non APGP [[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
#[[[[[[[[[[[[[[[[[[[[[[[[[ non APGP [[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        # if float(p1.fitness.values[0]) >= target:
        if float(p1.fitness.values[0]) >= target:
            #                print('Hi')
            if mettarget == 0:
                prev_gen = (len(population))*(gen-1)
                mettarget = counteval + prev_gen
                # print(f'GEN: {mettarget}, EVLN={counteval}')
                print(f'Target met: {mettarget}')
                print(f'Fitness: {float(p1.fitness.values[0])}')
                targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.fitness.values[0]), 'Met_at': mettarget}, index = {run})
            
                target_csv = f'{report_csv[:-4]}_Target.csv'
                #Export from dataframe to CSV file. Update if exists
                if os.path.isfile(target_csv):
                    targetmet_df.to_csv(target_csv, mode='a', header=False)
                else:
                    targetmet_df.to_csv(target_csv)                    
#MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
#[[[[[[[[[[[[[[[[[[[[[[[[[ non APGP [[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]  
#[[[[[[[[[[[[[[[[[[[[[[[[[ non APGP [[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END
                    
                    
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Identify an individual to be replaced - worst fitness
        #+++++++++++++++++++++++++++++++++++++++++++++
        #    p1, p2 = list(map(toolbox.clone, random.sample(population, 2)))
        #+++++++++++++++++++++++++++++++++++++++++++++
        #update_lock.acquire()          # LOCK !!!  
        # Identify a individual to replace from the population. Use Inverse Tournament
        candidates = selInverseTournament(population, k=1, tournsize=5)
        candidate = candidates[0]
        # Replace if offspring is better than candidate individual 
        if p1.fitness.values[0] > candidate.fitness.values[0]: # Max
        # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                population.append(p1) 
                population.remove(candidate)
        
        #update_lock.release()            # RELEASE !!!
        #+++++++++++++++++++++++++++++++++++++++++++++

        #Update hall of fame   ????==== INDENT TO HAPPEN ONLY IF A RELPLACEMENT IS DONE ====?????                                                                     
        try:
            halloffame.update(population)
        except AttributeError:
            pass  

    ################################################################################        
###############################################################################
    
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Collect Stats for the Generation 
    #+++++++++++++++++++++++++++++++++++++++++++++      
    def collectStatsGen():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
        #++++++++++ Collect Stats ++++++++++++++++++++
        record = stats.compile(population) if stats else {}
        logbook.record(run= run, gen=gen, nevals=counteval, **record)
        
        if verbose:
            print(logbook.stream) 
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Update hall of fame database for each generation
        hof_db.append([run, gen, halloffame[0].fitness.values[0], #halloffame[0].testfitness[0],
                       halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0])])# 
        # hof_db.append([run, gen, halloffame[0].fitness.values[0], halloffame[0].testfitness[0],
        #                halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0])])# 
                
        #+++++++ END OF GENERATION +++++++++++++++++++
        #+++++++++++++++++++++++++++++++++++++++++++++
        
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Create Report for the Run 
    #+++++++++++++++++++++++++++++++++++++++++++++        
    def collectStatsRun():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 

        #Put into dataframe
        chapter_keys = logbook.chapters.keys()
        sub_chaper_keys = [c[0].keys() for c in logbook.chapters.values()]
        
        data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                     in zip(sub_chaper_keys, logbook.chapters.values())]
        # data = np.array([[*a, *b, *c, *d] for a, b, c, d in zip(*data)])
        data = np.array([[*a, *b, *c] for a, b, c in zip(*data)])

        
        columns = reduce(add, [["_".join([x, y]) for y in s] 
                               for x, s in zip(chapter_keys, sub_chaper_keys)])
        df = pd.DataFrame(data, columns=columns)
        
        keys = logbook[0].keys()
        data = [[d[k] for d in logbook] for k in keys]
        for d, k in zip(data, keys):
            df[k] = d
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Export Report to local file
        if os.path.isfile(report_csv):
            df.to_csv(report_csv, mode='a', header=False)
        else:
            df.to_csv(report_csv)
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        ## Save 'Hall Of Fame' database
        #++++++++++++++++++++++++++++++++++++++++++++++
        #List to dataframe
        # hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
        hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Evln_time', 'Length', 'Best'])
        #Destination file (local)
        hof_csv = f'{report_csv[:-4]}_hof.csv'
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(hof_csv):
            hof_dframe.to_csv(hof_csv, mode='a', header=False)
        else:
            hof_dframe.to_csv(hof_csv)
###############################################################################


#+++++++++++++++++++++++++++++++++++++++++++++
#Create a Generation
#+++++++++++++++++++++++++++++++++++++++++++++
    # Begin the generational process
    for gen in range(1, ngen+1):
        # Generate offsprings -  equivalent to a generation / populations size
        poplnsize =  len(population)
#        poplnsize =  500
        counteval = 0 
        for h in range(poplnsize):
            breed()
        
        while counteval < poplnsize:
#            print(f'more pending')
            for j in range(poplnsize - counteval):
                breed()

        collectStatsGen()
    collectStatsRun()
    
###############################################################################       
    return population, logbook    
###############################################################################




#--------------------------------------------------
#       APGP
#--------------------------------------------------
"""
#==============================================================================
#==============================================================================
#++++++++++++++++++++++++++++++++++++++++++++++
# APGP_No_Gen_Stats - Parallel Steady State GP
#++++++++++++++++++++++++++++++++++++++++++++++
#==============================================================================
#==============================================================================
"""
def apgp2GenStats(population, toolbox, cxpb, mutpb, ngen, poolsize, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, datatrain=None, datatest=None, target =None):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    cxpb , mutpb = 0.9,0.1
    1/((cxpb + mutpb) - cxpb*mutpb)
    """  
    
    # get probability of evaluations and factor it in the number of breeds initiated
    # This will allow race to continue without stopping to check.
    factor = 1/((cxpb + mutpb) - cxpb*mutpb)
    #`````````````````````````````````````````````````````````````````````````````````````````````````````````  
    verbose = False
    gen=0
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
	#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    mettarget = 0 # 0 = not set
	#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END
	#``````````````````````````````````````````````````````````````````````````````      
    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    
    logbook = tools.Logbook()
    logbook.header = ['run', 'gen', 'nevals'] + (stats.fields if stats else [])

    #+++++++++++++++++++++++++++++++++++++++++++++
    #Evaluation of Initial Population
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Evaluate the individuals with an invalid fitness     
    counteval = 0
    
    for ind in population:
        if not ind.fitness.valid:
            # counteval =+ 1
            # xo, yo, zo, = toolbox.evaluate(ind, datatrain, datatest)
            xo, yo, = toolbox.evaluate(ind)
            ind.fitness.values = xo,          
            ind.evlntime = yo,
            # ind.testfitness = zo,

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    # logbook.record(gen=0, nevals=counteval, **record)
    if verbose:
        print(logbook.stream)

    gen=0
    hof_db=[]
    hof_db.append([run, gen, halloffame[0].fitness.values[0], #halloffame[0].testfitness[0], 
                   halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0])])
    # hof_db.append([run, gen, halloffame[0].fitness.values[0], halloffame[0].testfitness[0], 
    #                halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0])])
    

    # gen=ngen
    #`````````````````````````````````````````````````````````````````````````````````````````````````````````
    
    #+++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++
    from operator import attrgetter

    def selInverseTournament(individuals, k, tournsize, fit_attr="fitness"):
        """Select the worst individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for i in range(k):
            aspirants =  [random.choice(individuals) for i in range(tournsize)]
            chosen.append(min(aspirants, key=attrgetter(fit_attr)))
#            for l in aspirants: print(str(l.fitness))
        return chosen
        
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Breeding Function
    #+++++++++++++++++++++++++++++++++++++++++++++
    # define a breed function as nested.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, poolsize, mettarget

    # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
    # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
        """
        Modified to address (the error is occassional e.g. 1 in 15 runs):
		MemoryError: DEAP : Error in tree evaluation : Python cannot evaluate a tree higher than 90. 
        If error encountered try breeding again. 
		"""
#         successful = False
#         while successful == False:
#             try: 	
# #            print(f'more pending')
#                 #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))

        #++++++++ Crossover +++++++++++++++++++++++++++++++++++++
        if random.random() < cxpb:
            p1, p2 = toolbox.mate(p1, p2)
            del p1.fitness.values

        #++++++++ mutation on the offspring ++++++++++++++++               
        if random.random() < mutpb:
            p1, = toolbox.mutate(p1)
            del p1.fitness.values

        # Evaluate the offspring if it has changed
        if not p1.fitness.valid:

            # xo, yo, zo = toolbox.evaluate(p1, datatrain, datatest)
            xo, yo = toolbox.evaluate(p1)
            p1.evlntime = yo,
            # p1.testfitness = zo,
            p1.fitness.values = xo, 
            #Check if ZeroDivisionError, ValueError 
            if p1.fitness.values == (0.0101010101010101,) :
                p1.fitness.values = 0.0, #for maximising
            # if p1.testfitness == (0.0101010101010101,) :
            #     p1.testfitness = 0.0, #for maximising  
                # print('check this out')
                # print(str(p1))
                # print(str(p1.fitness.values))

            #++++++++ Counting evaluations +++++++++++++++++
            counteval_lock.acquire()
            counteval += 1 #Count the actual evaluations
            # counteval_lock.release()
            successful = True
            #++++++++ Counting evaluations +++++++++++++++++
            # counteval_lock.acquire()
            # counteval += 1 #Count the actual evaluations
            #         # # ````````````````````````````````````````````````````````````````````````````````````````````````````````
            if counteval % poplnsize == 0:
                genT = counteval/popsize
                
                # print(f'{counteval} evaluations initiated -- 	{round((100*counteval)/(ngen*poplnsize),2)}% of run {run}')
            # ````````````````````````````````````````````````````````````````````````````````````````````````````````
#                    # counteval_lock.release()
#                        collectStatsGen(gen)					
#                    counteval_lock.release()
                counteval_lock.release()
                collectStatsGen(genT)

            else: counteval_lock.release()

            # except MemoryError :
            #     print('MemoryError encountered')
            #     successful = False
        # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
        # 000000000000000000000000000000000000000000000000000000000000000000000000000000000

                
		#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
		#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        # if float(p1.fitness.values[0]) >= target:
        if float(p1.fitness.values[0]) >= target:
            #                print('Hi')
            if mettarget == 0:
                mettarget = counteval
                print(f'Target met: {counteval}')
                print(f'Fitness: {float(p1.fitness.values[0])}')
                targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.fitness.values[0]), 'Met_at': mettarget}, index = {run})
            
                target_csv = f'{report_csv[:-4]}_Target.csv'
                #Export from dataframe to CSV file. Update if exists
                if os.path.isfile(target_csv):
                    targetmet_df.to_csv(target_csv, mode='a', header=False)
                else:
                    targetmet_df.to_csv(target_csv)                    
		#MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
		#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END

        #+++++++++++++++++++++++++++++++++++++++++++++
#       Identify an individual to be replaced - worst fitness
        #+++++++++++++++++++++++++++++++++++++++++++++
#            p1, p2 = list(map(toolbox.clone, random.sample(population, 2)))
        #+++++++++++++++++++++++++++++++++++++++++++++
        update_lock.acquire()          # LOCK !!!  
        # Identify a individual to replace from the population. Use Inverse Tournament
        candidates = selInverseTournament(population, k=1, tournsize=5)
        candidate = candidates[0]
        # Replace if offspring is better than candidate individual 
        if p1.fitness.values[0] > candidate.fitness.values[0]: # Max
        # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                population.append(p1) 
                population.remove(candidate)
        
        update_lock.release()            # RELEASE !!!
        #+++++++++++++++++++++++++++++++++++++++++++++

#    Update hall of fame   ????==== INDENT TO HAPPEN ONLY IF A RELPLACEMENT IS DONE ====?????                                                                     
        try:
            halloffame.update(population)
        except AttributeError:
            pass  

    ###############################################################################
    
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Collect Stats for the Generation 
    #+++++++++++++++++++++++++++++++++++++++++++++      
    def collectStatsGen(gen):
        nonlocal population, stats, run, counteval, logbook, verbose, hof_db, halloffame, report_csv #, gen
        #++++++++++ Collect Stats ++++++++++++++++++++
        record = stats.compile(population) if stats else {}
        logbook.record(run= run, gen=gen, nevals=counteval, **record)
        
        if verbose:
            print(logbook.stream) 
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Update hall of fame database for each generation
        hof_db.append([run, gen, halloffame[0].fitness.values[0], #halloffame[0].testfitness[0], 
                       halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0]) ])# 
        # hof_db.append([run, gen, halloffame[0].fitness.values[0], halloffame[0].testfitness[0], 
        #                halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0]) ])# 
        
    #+++++++ END OF GENERATION +++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++
        
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Create Report for the Run 
    #+++++++++++++++++++++++++++++++++++++++++++++        
    def collectStatsRun():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 

        #Put into dataframe
        chapter_keys = logbook.chapters.keys()
        sub_chaper_keys = [c[0].keys() for c in logbook.chapters.values()]
        
        data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                     in zip(sub_chaper_keys, logbook.chapters.values())]
        # data = np.array([[*a, *b, *c, *d] for a, b, c, d in zip(*data)])
        data = np.array([[*a, *b, *c] for a, b, c in zip(*data)])

        
        columns = reduce(add, [["_".join([x, y]) for y in s] 
                               for x, s in zip(chapter_keys, sub_chaper_keys)])
        df = pd.DataFrame(data, columns=columns)
        
        keys = logbook[0].keys()
        data = [[d[k] for d in logbook] for k in keys]
        for d, k in zip(data, keys):
            df[k] = d
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Export Report to local file
        if os.path.isfile(report_csv):
            df.to_csv(report_csv, mode='a', header=False)
        else:
            df.to_csv(report_csv)
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        ## Save 'Hall Of Fame' database
        #++++++++++++++++++++++++++++++++++++++++++++++
        #List to dataframe
        # hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
        hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Evln_time', 'Length', 'Best'])
        #Destination file (local)
        hof_csv = f'{report_csv[:-4]}_hof.csv'
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(hof_csv):
            hof_dframe.to_csv(hof_csv, mode='a', header=False)
        else:
            hof_dframe.to_csv(hof_csv)
        ###############################################################################


# NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW 
##+++++++++++++++++++++++++++++++++++++++++++++
##Create a Generation
##+++++++++++++++++++++++++++++++++++++++++++++
#    # Begin the generational process   
#    for gen in range(1, ngen+1):

    tp = ThreadPool(poolsize)  # <-------------------------------------------- (3a)
        # Generate offsprings -  equivalent to a generation based on populations size
    poplnsize =  len(population)
    targetevalns = poplnsize*ngen
#        poplnsize =  500
    
    counteval = 0 
    # while (counteval < poplnsize+1): 
    # for h in range(int(poplnsize*ngen)):
    for h in range(int(poplnsize*ngen*factor)):
#            print(counteval)
        tp.apply_async(breed)

#   Append the current generation statistics to the logbook
    tp.close() # <---------------------------------------??????????????
    tp.join() #  <---------------------------------------??????????????
    
    # If last generation is not complete do a few more breed operations
    while counteval < targetevalns:
        #print(f'more pending')
        tp = ThreadPool(poolsize)  # <---------------------------------------- (3b)
#if count < target:psize
        for j in range(targetevalns - counteval):
            tp.apply_async(breed)
        tp.close() # <---------------------------------------??????????????
        tp.join() #  <---------------------------------------??????????????


    print(f'done  : {counteval}')
    print(f'Target: {targetevalns}')
    print(threading.active_count())

	
    # collectStatsGen()
    collectStatsRun()
# `````````````````````````````````````````````````````````````````````````````   
    
###############################################################################
###############################################################################       
    return population, logbook    
###############################################################################

"""   
#==============================================================================
#============================================================================== 
#==============================================================================
#==============================================================================
"""
#--------------------------------------------------
#       APGP
#--------------------------------------------------
"""
#==============================================================================
#==============================================================================
#++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++
# APGP_No_Gen_Stats - Parallel Steady State GP
#++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++
#==============================================================================
#==============================================================================
"""
def apgp2GenStatsL(population, toolbox, cxpb, mutpb, ngen, poolsize, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, datatrain=None, datatest=None, target =None):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    cxpb , mutpb = 0.9,0.1
    1/((cxpb + mutpb) - cxpb*mutpb)
    """  
    
    # get probability of evaluations and factor it in the number of breeds initiated
    # This will allow race to continue without stopping to check.
    factor = 1/((cxpb + mutpb) - cxpb*mutpb)
    #`````````````````````````````````````````````````````````````````````````````````````````````````````````  
    verbose = False
    gen=0
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
	#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    mettarget = 0 # 0 = not set
	#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END
	#``````````````````````````````````````````````````````````````````````````````      
    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    
    logbook = tools.Logbook()
    logbook.header = ['run', 'gen', 'nevals'] + (stats.fields if stats else [])

    #+++++++++++++++++++++++++++++++++++++++++++++
    #Evaluation of Initial Population
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Evaluate the individuals with an invalid fitness     
    counteval = 0
    
    for ind in population:
        if not ind.fitness.valid:
            # counteval =+ 1
            # xo, yo, zo, = toolbox.evaluate(ind, datatrain, datatest)
            xo, yo, = toolbox.evaluate(ind)
            ind.fitness.values = xo,          
            ind.evlntime = yo,
            # ind.testfitness = zo,

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    # logbook.record(gen=0, nevals=counteval, **record)
    if verbose:
        print(logbook.stream)

    gen=0
    hof_db=[]
    hof_db.append([run, gen, halloffame[0].fitness.values[0], #halloffame[0].testfitness[0], 
                   halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0])])
    # hof_db.append([run, gen, halloffame[0].fitness.values[0], halloffame[0].testfitness[0], 
    #                halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0])])
    

    # gen=ngen
    #`````````````````````````````````````````````````````````````````````````````````````````````````````````
    
    #+++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++
    from operator import attrgetter

    def selInverseTournament(individuals, k, tournsize, fit_attr="fitness"):
        """Select the worst individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for i in range(k):
            aspirants =  [random.choice(individuals) for i in range(tournsize)]
            chosen.append(min(aspirants, key=attrgetter(fit_attr)))
#            for l in aspirants: print(str(l.fitness))
        return chosen
        
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Breeding Function
    #+++++++++++++++++++++++++++++++++++++++++++++
    # define a breed function as nested.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, poolsize, mettarget

    # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
    # 000000000000000000000000000000000000000000000000000000000000000000000000000000000
        """
        Modified to address (the error is occassional e.g. 1 in 15 runs):
		MemoryError: DEAP : Error in tree evaluation : Python cannot evaluate a tree higher than 90. 
        If error encountered try breeding again. 
		"""
#         successful = False
#         while successful == False:
#             try: 	
# #            print(f'more pending')
#                 #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))

        #++++++++ Crossover +++++++++++++++++++++++++++++++++++++
        if random.random() < cxpb:
            p1, p2 = toolbox.mate(p1, p2)
            del p1.fitness.values

        #++++++++ mutation on the offspring ++++++++++++++++               
        if random.random() < mutpb:
            p1, = toolbox.mutate(p1)
            del p1.fitness.values

        # Evaluate the offspring if it has changed
        if not p1.fitness.valid:

            # xo, yo, zo = toolbox.evaluate(p1, datatrain, datatest)
            xo, yo = toolbox.evaluate(p1)
            p1.evlntime = yo,
            # p1.testfitness = zo,
            p1.fitness.values = xo, 
            #Check if ZeroDivisionError, ValueError 
            if p1.fitness.values == (0.0101010101010101,) :
                p1.fitness.values = 0.0, #for maximising
            # if p1.testfitness == (0.0101010101010101,) :
            #     p1.testfitness = 0.0, #for maximising  
                # print('check this out')
                # print(str(p1))
                # print(str(p1.fitness.values))

            #++++++++ Counting evaluations +++++++++++++++++
            counteval_lock.acquire()
            counteval += 1 #Count the actual evaluations
            # counteval_lock.release()
            successful = True
            #++++++++ Counting evaluations +++++++++++++++++
            # counteval_lock.acquire()
            # counteval += 1 #Count the actual evaluations
            #         # # ````````````````````````````````````````````````````````````````````````````````````````````````````````
            if counteval % poplnsize == 0:
                genT = counteval/popsize
                
                # print(f'{counteval} evaluations initiated -- 	{round((100*counteval)/(ngen*poplnsize),2)}% of run {run}')
            # ````````````````````````````````````````````````````````````````````````````````````````````````````````
            # ```````````````````````````````````````````
                
            #================================================================== 
                #0000000000000000000000000000000000000000000000000000000000000
            #==================================================================   
                
            # `````````````` Option 1 - Lock Population before taking generational stats ````````````````````
                    # counteval_lock.release()
                collectStatsGen(genT)					
            counteval_lock.release()
                    
            # `````````````` Option 2 - Do not Lock Population before taking generational stats `````````````
            #     counteval_lock.release()
            #     collectStatsGen(genT)

            # else: counteval_lock.release()

            #================================================================== 
                #0000000000000000000000000000000000000000000000000000000000000
            #==================================================================   

		#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
		#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        # if float(p1.fitness.values[0]) >= target:
        if float(p1.fitness.values[0]) >= target:
            #                print('Hi')
            if mettarget == 0:
                mettarget = counteval
                print(f'Target met: {counteval}')
                print(f'Fitness: {float(p1.fitness.values[0])}')
                targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.fitness.values[0]), 'Met_at': mettarget}, index = {run})
            
                target_csv = f'{report_csv[:-4]}_Target.csv'
                #Export from dataframe to CSV file. Update if exists
                if os.path.isfile(target_csv):
                    targetmet_df.to_csv(target_csv, mode='a', header=False)
                else:
                    targetmet_df.to_csv(target_csv)                    
		#MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
		#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END

        #+++++++++++++++++++++++++++++++++++++++++++++
#       Identify an individual to be replaced - worst fitness
        #+++++++++++++++++++++++++++++++++++++++++++++
#            p1, p2 = list(map(toolbox.clone, random.sample(population, 2)))
        #+++++++++++++++++++++++++++++++++++++++++++++
        update_lock.acquire()          # LOCK !!!  
        # Identify a individual to replace from the population. Use Inverse Tournament
        candidates = selInverseTournament(population, k=1, tournsize=5)
        candidate = candidates[0]
        # Replace if offspring is better than candidate individual 
        if p1.fitness.values[0] > candidate.fitness.values[0]: # Max
        # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                population.append(p1) 
                population.remove(candidate)
        
        update_lock.release()            # RELEASE !!!
        #+++++++++++++++++++++++++++++++++++++++++++++

#    Update hall of fame   ????==== INDENT TO HAPPEN ONLY IF A RELPLACEMENT IS DONE ====?????                                                                     
        try:
            halloffame.update(population)
        except AttributeError:
            pass  

    ###############################################################################
    
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Collect Stats for the Generation 
    #+++++++++++++++++++++++++++++++++++++++++++++      
    def collectStatsGen(gen):
        nonlocal population, stats, run, counteval, logbook, verbose, hof_db, halloffame, report_csv #, gen
        #++++++++++ Collect Stats ++++++++++++++++++++
        record = stats.compile(population) if stats else {}
        logbook.record(run= run, gen=gen, nevals=counteval, **record)
        
        if verbose:
            print(logbook.stream) 
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Update hall of fame database for each generation
        hof_db.append([run, gen, halloffame[0].fitness.values[0], #halloffame[0].testfitness[0], 
                       halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0]) ])# 
        # hof_db.append([run, gen, halloffame[0].fitness.values[0], halloffame[0].testfitness[0], 
        #                halloffame[0].evlntime[0], len(halloffame[0]), str(halloffame[0]) ])# 
        
    #+++++++ END OF GENERATION +++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++
        
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Create Report for the Run 
    #+++++++++++++++++++++++++++++++++++++++++++++        
    def collectStatsRun():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 

        #Put into dataframe
        chapter_keys = logbook.chapters.keys()
        sub_chaper_keys = [c[0].keys() for c in logbook.chapters.values()]
        
        data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                     in zip(sub_chaper_keys, logbook.chapters.values())]
        # data = np.array([[*a, *b, *c, *d] for a, b, c, d in zip(*data)])
        data = np.array([[*a, *b, *c] for a, b, c in zip(*data)])

        
        columns = reduce(add, [["_".join([x, y]) for y in s] 
                               for x, s in zip(chapter_keys, sub_chaper_keys)])
        df = pd.DataFrame(data, columns=columns)
        
        keys = logbook[0].keys()
        data = [[d[k] for d in logbook] for k in keys]
        for d, k in zip(data, keys):
            df[k] = d
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Export Report to local file
        if os.path.isfile(report_csv):
            df.to_csv(report_csv, mode='a', header=False)
        else:
            df.to_csv(report_csv)
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        ## Save 'Hall Of Fame' database
        #++++++++++++++++++++++++++++++++++++++++++++++
        #List to dataframe
        # hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
        hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Evln_time', 'Length', 'Best'])
        #Destination file (local)
        hof_csv = f'{report_csv[:-4]}_hof.csv'
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(hof_csv):
            hof_dframe.to_csv(hof_csv, mode='a', header=False)
        else:
            hof_dframe.to_csv(hof_csv)
        ###############################################################################


# NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW 
##+++++++++++++++++++++++++++++++++++++++++++++
##Create a Generation
##+++++++++++++++++++++++++++++++++++++++++++++
#    # Begin the generational process   
#    for gen in range(1, ngen+1):

    tp = ThreadPool(poolsize)  # <-------------------------------------------- (3a)
        # Generate offsprings -  equivalent to a generation based on populations size
    poplnsize =  len(population)
    targetevalns = poplnsize*ngen
#        poplnsize =  500
    
    counteval = 0 
    # while (counteval < poplnsize+1): 
    # for h in range(int(poplnsize*ngen)):
    for h in range(int(poplnsize*ngen*factor)):
#            print(counteval)
        tp.apply_async(breed)

#   Append the current generation statistics to the logbook
    tp.close() # <---------------------------------------??????????????
    tp.join() #  <---------------------------------------??????????????
    
    # If last generation is not complete do a few more breed operations
    while counteval < targetevalns:
        #print(f'more pending')
        tp = ThreadPool(poolsize)  # <---------------------------------------- (3b)
#if count < target:psize
        for j in range(targetevalns - counteval):
            tp.apply_async(breed)
        tp.close() # <---------------------------------------??????????????
        tp.join() #  <---------------------------------------??????????????


    print(f'done  : {counteval}')
    print(f'Target: {targetevalns}')
    print(threading.active_count())

	

    # collectStatsGen()
    collectStatsRun()
# `````````````````````````````````````````````````````````````````````````````   
    
###############################################################################
###############################################################################       
    return population, logbook    
###############################################################################

"""   
#==============================================================================
#============================================================================== 
#==============================================================================
#==============================================================================
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++

def lastgenstats(population, toolbox, gen=0,  run=0, report_csv=None):
#    nonlocal population, toolbox, report_csv, run, gen
    lastgen_db=[]    
    for j in range(len(population)):
        # xo, yo, zo = toolbox.evaluate(population[j], datatrain, datatest)
        xo, yo = toolbox.evaluate(population[j])#, datatrain, datatest)
        population[j].fitness.values = xo,
        population[j].evlntime = yo,
        # population[j].testfitness = zo,
        lastgen_db.append([run, gen, float(str(population[j].fitness)[1:-2]), float(str(population[j].evlntime)[1:-2]), len(population[j]), str(population[j])])
    lastgen_dframe=pd.DataFrame(lastgen_db, columns=['Run', 'Generation', 'Train_Fitness', 'Evln_time', 'Length', 'Best'])
    #     lastgen_db.append([run, gen, float(str(population[j].fitness)[1:-2]), float(str(population[j].testfitness)[1:-2]), float(str(population[j].evlntime)[1:-2]), len(population[j]), str(population[j])])
    # lastgen_dframe=pd.DataFrame(lastgen_db, columns=['Run', 'Generation', 'Train_Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
    
    #Destination file
    lastgen_csv = f'{report_csv[:-4]}_lastgen.csv'
    #Export from dataframe to CSV file. Update if exists
    if os.path.isfile(lastgen_csv):
        lastgen_dframe.to_csv(lastgen_csv, mode='a', header=False)
    else:
        lastgen_dframe.to_csv(lastgen_csv)

# """
# ============================================================================
# ============================================================================
# """
      

def main():
    random.seed(2020)
    
    # cxpb, mutpb, ngen = 0.7, 0.3, 50
    # trgt=0.75
    # with  open("ant/santafe_trail.txt") as trail_file:
    #   ant.parse_matrix(trail_file)    

      #==============================
      #       STEADY-STATE
      #==============================
    tag = f'Ph6_SSGP_prog-{PARITY_FANIN_M}{Prob}'#Standard GP - Steady State
    report_csv =  f"{result_dir}{run_time}_{system}_{tag}.csv"    
   
    for i in range(1, runs+1):
    # for i in range(runs+1):
        run = i
        print(f'\n Run {run} of {tag}')
        pop = toolbox.population(n=popsize)
#        population = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
      
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
        ##stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime)#, testfitness=stats_testfitness)
        # mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness)
  
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        gpSteadyState(pop, toolbox, cxpb, mutpb, ngen, mstats, halloffame=hof, run=run, report_csv=report_csv, target =trgt)
        # gpSteadyState(pop, toolbox, cxpb, mutpb, ngen, mstats, halloffame=hof, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target =trgt)
        # algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats, halloffame=hof)    
 
        print('Taking stats for the last generation....')

        lastgenstats(pop, toolbox, gen=ngen, run=run, report_csv=report_csv)#, datatrain=datatrain, datatest=datatestGEN....??  (9b)



    #==============================
    #           APGP              #
    #==============================
    poolsizelist = [5, 25, 50, 75, 100]
    # poolsizelist = [25]#, 50, 75, 100]    
    for poolsize in poolsizelist:
        tag = f'Ph6_APGP_ps{poolsize}_prog-{PARITY_FANIN_M}{Prob}'#   APGP
        report_csv =  f"{result_dir}{run_time}_{system}_{tag}.csv"    
        # poolsize = 25
        for i in range(1, runs+1):
        # for i in range(runs+1):
            run = i
            print(f'\n Run {run} of {tag}')
            pop = toolbox.population(n=popsize)
            hof = tools.HallOfFame(1)
            
            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
            stats_size = tools.Statistics(len)
            stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
            ##stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
            mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime)#, testfitness=stats_testfitness)
            # mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness)
        
            mstats.register("avg", numpy.mean)
            mstats.register("std", numpy.std)
            mstats.register("min", numpy.min)
            mstats.register("max", numpy.max)
            # apgp2GenStats(pop, toolbox, cxpb, mutpb, ngen, poolsize, mstats, halloffame=hof, verbose=__debug__, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target =trgt)   
            apgp2GenStats(pop, toolbox, cxpb, mutpb, ngen, poolsize, mstats, halloffame=hof, verbose=__debug__, run=run, report_csv=report_csv, target =trgt)
            print('Taking stats for the last generation....')
        
            lastgenstats(pop, toolbox, gen=ngen, run=run, report_csv=report_csv)#, datatrain=datatrain, datatest=datatestGEN....??  (9b)

     #==============================
     # STEADY-STATE (BLOAT CONTROL)
     #==============================
    tag = f'Ph6_SSGP-BC_prog-{PARITY_FANIN_M}{Prob}'#Standard GP - Steady State
#    report_csv = f"C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Phase3\\{run_time}_{tag}.csv"
    report_csv =  f"{result_dir}{run_time}_{system}_{tag}.csv"    

    for i in range(1, runs+1):
    # for i in range(runs+1):
        run = i
        print(f'\n Run {run} of {tag}')
        pop = toolbox.population(n=popsize)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
        #stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime)#, testfitness=stats_testfitness)
        # mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness)

        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        # gpDoubleT(pop, toolbox, cxpb, mutpb, ngen, mstats, halloffame=hof, verbose=__debug__, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target =trgt)
        gpDoubleT(pop, toolbox, cxpb, mutpb, ngen, mstats, halloffame=hof, verbose=__debug__, run=run, report_csv=report_csv, target =trgt)

        print('Taking stats for the last generation....')

        lastgenstats(pop, toolbox, gen=ngen, run=run, report_csv=report_csv)#, datatrain=datatrain, datatest=datatestGEN....??  (9b)

    #==============================
    #           APGP   LOCK           #
    #==============================
    poolsizelist = [5, 25, 50, 75, 100]
    # poolsizelist = [25]#, 50, 75, 100]    
    for poolsize in poolsizelist:
        tag = f'Ph6_APGP_ps{poolsize}_prog-{PARITY_FANIN_M}L{Prob}'#   APGP
        report_csv =  f"{result_dir}{run_time}_{system}_{tag}.csv"    
        # poolsize = 25
        for i in range(1, runs+1):
        # for i in range(runs+1):
            run = i
            print(f'\n Run {run} of {tag}')
            pop = toolbox.population(n=popsize)
            hof = tools.HallOfFame(1)
            
            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
            stats_size = tools.Statistics(len)
            stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
            ##stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
            mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime)#, testfitness=stats_testfitness)
            # mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness)
        
            mstats.register("avg", numpy.mean)
            mstats.register("std", numpy.std)
            mstats.register("min", numpy.min)
            mstats.register("max", numpy.max)
            # apgp2GenStats(pop, toolbox, cxpb, mutpb, ngen, poolsize, mstats, halloffame=hof, verbose=__debug__, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target =trgt)   
            apgp2GenStatsL(pop, toolbox, cxpb, mutpb, ngen, poolsize, mstats, halloffame=hof, verbose=__debug__, run=run, report_csv=report_csv, target =trgt)
            print('Taking stats for the last generation....')
        
            lastgenstats(pop, toolbox, gen=ngen, run=run, report_csv=report_csv)#, datatrain=datatrain, datatest=datatestGEN....??  (9b)


#==============================================================================
if __name__ == "__main__":
    main()    
#==============================================================================

    
"""
for i in range(30,70):
    print(str(pop[i]))

"""
