# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:23:20 2020

@author: aliyus
"""
# 20 GENERATIONS AND HOF RE-EVALUATED TO GET A COMPARABLE EVALUATION TIME


# -------------------------------------

# dataPLMB='banknote1.csv' #  - banknote1 (class -1: 762, 1: 609 (44%); inst: 1371)
# dataPLMB='bloodtransfusion1.csv' #(class - 1: 570, 2:	177 (24%); inst: 747)
# dataPLMB='diabetes1.csv' # diabetes1 (class - 0: 500, 1: 267 (35%); inst: 767)
# dataPLMB = 'IndianLiverDis.csv'
dataPLMB = 'BreastCancerWins1.csv'
# dataPLMB='ozone1.csv'
# dataPLMB='phoneme1.csv' #  - phoneme1 (class - 1: 3818, 2: 1586 (29%); inst: 5404)
# dataPLMB='spambase1.csv' # (class - 0: 2788, 1: 1812 (39); inst: 4,600)

settarget=0.9

# dataPLMB='mushroom1.csv'
# dataPLMB='Note.txt'
# dataPLMB='covtype.csv' # Covertype -7 classes (1=36%, 2=49%, 3=6%, 4=0%, 5=2%, 6=3%, 7=4%); 54 variable, inst: 581,012)
# dataPLMB='ozone2.csv'

# # ---------------------------------------------------------------------------
#   os.chdir('C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Ph5\\')

"""
# ========= Time Control Mode =================================================
# =============================================================================
"""
# ----------------- Choose options --------------------------------------------
PREDSYS = 'REGRESSION'

PREDSYS = 'CLASSIFICATION'


ftStd = 'yes'#
# ftStd = 'no'

ftSC = 'yes'
# ftSC = 'no'#

ftTC1 = 'yes'#
# ftTC1 = 'no'

APGP = 'yes'
# APGP = 'no'#

ftAdj = 'yes'#
ftAdj = 'no'

stdGP = 'yes'
stdGP = 'no'#

# -----------------------------------------------------------------------------
ftTCAR2 = 'yes'#
ftTCAR2 = 'no'

APGPAR2 = 'yes'#
APGPAR2 = 'no'
# -----------------------------------------------------------------------------

# Create list with Methods to Run
methodlist = []

try:
    if ftStd == 'yes': methodlist.append('ftStd')
except Exception:     pass

try:
    if ftSC == 'yes': methodlist.append('ftSC')
except Exception:     pass

try:
    if ftTC1 == 'yes': methodlist.append('ftTC1')
except Exception:     pass

try:
    if APGP == 'yes': methodlist.append('APGP')
except Exception:     pass

try:
    if stdGP == 'yes': methodlist.append('stdGP')
except Exception:     pass

try:
    if ftAdj == 'yes': methodlist.append('ftAdj')
except Exception:     pass

# -----------------------------------------------------------------------------
try:
    if ftTCAR2 == 'yes': methodlist.append('ftTCAR2')
except Exception:     pass

try:
    if APGPAR2 == 'yes': methodlist.append('APGPAR2')
except Exception:     pass

# ----------------------------------------------------------------------------- 
popsize = 500
runs = 50 #1 # 
nofGen = 20
THREADLIST = [100,25,250]#, [50,100] # 
FLISize = 10
# ----------------------------------------------------------------------------- 

# # !!!!!!! temporary settings for testing only !!!!!!
# popsize = 100 # 
# runs = 1
# nofGen = 10 # 50 # 
# THREADLIST = [50] # [25,100,250]#, 
# # ----------------------------------------------------------------------------- 

# Initialise
import csv
import itertools
import operator
import math
import random
import numpy
from deap import base
from deap import creator
from deap import tools
from deap import gp
import datetime
import time
#from . import tools
import pandas as pd
import numpy as np
import os
from functools import reduce
from operator import add, itemgetter
#from math import exp, cos, sin, log
from multiprocessing.pool import ThreadPool, threading
from deap_pX.apgp import gpDoubleT2, gpSteadyState#gpDoubleTcx#,apgpwGenStats, gpDoubleT,  lastgenstats, apgpNoGenStats, gpDoubleTFL, gpDoubleTFL2
from deap_pX import tctools
#from multiprocessing.pool import ThreadPool, threading
import matplotlib.pyplot as plt
import networkx as nx
#import pydot
#from networkx.drawing.nx_pydot import graphviz_layout
#------------------------------------------------------------------------------
run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M") #           (2)
#===============================================================================


#################################################################################
try:
    devicename = os.environ['COMPUTERNAME']
    if devicename == 'DESKTOP-MLNSBQ2':
        system = 'laptop'
    elif devicename == 'DESKTOP-P61PME5':
        system = 'bsystem'
    elif devicename == 'DESKTOP-JAN9GCB':
        system =    'NBOOK'
    elif devicename == 'DESKTOP-4VA0QI6':
        system = 'Dktop2' 
    else: system = 'desktop'

except KeyError:
    system = 'server'  
#==============================================================================
#==============================================================================
if system == 'server':
    datafolder = f"/home/aliyu/Documents/Features/newdata/{dataPLMB}"
elif system == 'laptop':
    datafolder = f"C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Ph6\\classfnData\\{dataPLMB}"
elif system == 'desktop':
    datafolder = f"C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Ph6\\classfnData\\{dataPLMB}"
elif system == 'bsystem':
    datafolder = f"C:\\Users\\PC\\Documents\\newdata\\{dataPLMB}"
elif system == 'NBOOK':
    datafolder = f"C:\\Users\\Aliyu Sambo\\OneDrive - Birmingham City University\\Experiment_Ph6\\classfnData\\{dataPLMB}"
elif system == 'Dktop2':
    datafolder = f"C:\\Users\\user\\Documents\\02_FEATURES\\classificationGPML\\classfnData\\{dataPLMB}" #classificationGPML\classfnData
#################################################################################



# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
with open(f"{data_dir}{data}.csv") as train:
#with open("C:\\Users\\user\\Documents\\Ph6\\diabetes1.csv") as train:
    #with open("spambase.csv") as spambase:
    trainReader = csv.reader(train)
    Tpoints = list(list(float(item) for item in row) for row in trainReader)

#split data: random selection without replacement
#Tpoints = points.copy()
random.seed(2021)
if data == 'spambase1' or data == 'ozone2' or data == 'BreastCancerWins1' or data == 'diabetes1' :
    random.seed(2019)# a temporary workaround to ensure a similar class distribution between training and test dataset -----???????????
x1=random.shuffle(Tpoints)   
split = int(len(Tpoints)*0.2)
datatrain=Tpoints[split:len(Tpoints)]
datatest=Tpoints[0:split]

trgY = len(Tpoints[1])-1
# --------------------------------------
train0 = []
train1 = []
for i in range(len(datatrain)):
    if datatrain[i][trgY]==0:
        train0.append(datatrain[i])
# for i in range(len(datatrain)):
    elif datatrain[i][trgY]==1:
        train1.append(datatrain[i])
    else: print('class of data instances not 0 or 1')
# --------------------------------------
test0 = []
test1 = []
for i in range(len(datatest)):
    if datatest[i][trgY]==0:
        test0.append(datatest[i])
    elif datatest[i][trgY]==1:
        test1.append(datatest[i])
    else: print('class of data instances not 0 or 1')
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------




# #===========================================================================
# from sklearn.datasets import load_breast_cancer
# # data = load_breast_cancer()
# Xdata, ydata = load_breast_cancer(return_X_y=True)

# from sklearn.datasets import fetch_rcv1
# data = fetch_rcv1()

# from sklearn.datasets import fetch_20newsgroups_vectorized
# Xdata, ydata = fetch_20newsgroups_vectorized(return_X_y=True)

# from sklearn.datasets import fetch_covtype
# Xdata, ydata = fetch_covtype(return_X_y=True)
# #===========================================================================

def div(left, right):
    return left / right

"""
==============================================================================
                EVALUATION FUNCTIONS
==============================================================================
"""
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load Classification the algorithms:
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing


""" 
==============================================================================
=======                     OPERATORS                               ==========
==============================================================================
"""

######################################
#        GP Crossovers               #
######################################
def cxOnePointFt(ind1, ind2):
    """Randomly select in each individual and exchange each subtree with the
    point as root between each individual.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    count = 0
    success = False
    while count < 10 and success == False:
        count += 1
        index1 = random.randrange(1,len(ind1)) # Leave the root node 
        index2 = random.randrange(1,len(ind2)) # Leave the root node        
        # CHECK IF SUBTREE EXCHANGE WILL LEAD TO VALID INDIVIDUALS        
        #1.  Check index types ================================================
        # ---- check ind1 ----------------
        nodes, edges, labels = gp.graph(ind1) 
        if labels[index1] == 'feature':
            pointtype1 = 'feature' 
        else:
            pointtype1 = 'non-feature'
        #print(f'pointtype1: {pointtype1}')
        # ---- check parent1
        pointparent1 = ''
        if pointtype1 != 'feature':
            #check parenttype
            #print(edges[index1-1])
            parent1 = edges[index1-1][0]
            pointparent1 = labels[parent1]
            #print(f'Parent: {pointparent1}')
        # ---- check ind2 ----------------
        nodes, edges, labels = gp.graph(ind2) 
        if labels[index2] == 'feature':
            pointtype2 = 'feature' 
        else:
            pointtype2 = 'non-feature'  
        #print(f'pointtype2: {pointtype2}')
        # ---- check parent2 
        pointparent2 = ''
        if pointtype2 != 'feature':
            #check parenttype
            #print(edges[index2-1])
            parent1 = edges[index2-1][0]
            pointparent2 = labels[parent1]  
            #print(f'Parent: {pointparent2}')
        # =====================================================================
        #2.  If condition to produce valid offsprings are met proceed
        if (pointtype1 == pointtype2):
            slice1 = ind1.searchSubtree(index1)
            slice2 = ind2.searchSubtree(index2)
            ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
            success = True
        # If  FUNC is replacing non-FUNC then ensure that parent of the non-FUNC is a FUNC
        if (pointtype1 != pointtype2):
            if (pointtype1 == 'feature' and pointtype2 == 'non-feature' and pointparent2 == 'feature'):
                slice1 = ind1.searchSubtree(index1)
                slice2 = ind2.searchSubtree(index2)
                ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
                success = True
            # if the reverse is met - (easy reading)
            elif (pointtype2 == 'feature' and pointtype1 == 'non-feature' and pointparent1 == 'feature'):
                slice1 = ind1.searchSubtree(index1)
                slice2 = ind2.searchSubtree(index2)
                ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
                success = True
        #print(f'successful crossover = {success}')
        #print(f'attempts = {count}')
    return ind1, ind2


## -----------------------------------------------------------------------------
"""
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           FEATURES: APGP - (NMSE or AdjR2)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
#from multiprocessing.pool import ThreadPool, threading
def APGPFtL(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, poolsize=25, datatrain=None, datatest=None, target =None, poplnType =None,FtEvlType =None, pset =None):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
    factor = 1/((cxpb + mutpb) - cxpb*mutpb)

    counteval = 0
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    mettarget = 0 # 0 = not set
	#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END
	#``````````````````````````````````````````````````````````````````````````````  
    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    # -------------------------------------------------------------------------
    logbook = tools.Logbook()
    logbook.header = ['run', 'gen', 'nevals'] + (stats.fields if stats else [])
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++
    # Evaluation of Initial Population  (NMSE or AdjR2)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++
    for ind in population:
        if poplnType == 'Features' and FtEvlType == 'AdjR2':
            if not ind.fitness.valid:
                xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(ind, datatrain, datatest)
                ind.evlntime = yo,
                ind.testfitness = zo,
                ind.fitness.values = xo,
                ind.nooffeatures = noft, 
                ind.mser2_train = MSER2_train, 
                ind.mser2_test  = MSER2_test,             

                if ind.fitness.values == (0.0101010101010101,) :
                    ind.fitness.values = 0.0, #for maximising
                if ind.testfitness == (0.0101010101010101,) :
                    ind.testfitness = 0.0, #for maximising         

        elif poplnType == 'Features' and FtEvlType == 'MSE':
                xo, yo, zo, noft = toolbox.evaluate(ind, datatrain, datatest)
                ind.evlntime = yo,
                ind.testfitness = zo,
                ind.fitness.values = xo,
                ind.nooffeatures = noft,         

                if ind.fitness.values == (0.0101010101010101,) :
                    ind.fitness.values = 0.0, #for maximising
                if ind.testfitness == (0.0101010101010101,) :
                    ind.testfitness = 0.0, #for maximising   
    #+++++++++++++++++++++++++++++++++++++++++++++
    try:
        halloffame.update(population)
    except AttributeError:
        pass
    #+++++++++++++++++++++++++++++++++++++++++++++
    record = stats.compile(population) if stats else {}
    logbook.record(run=run, gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)  
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Capture best individual for Generation 0
    gen=0
    hof_db=[]

    # -------------------------------------------------------------------------
    # Collect HOF Data  (NMSE or AdjR2) 
    if poplnType == 'Features' and FtEvlType == 'AdjR2':
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), halloffame[0].evlntime, len(halloffame[0]),
                       int(str(halloffame[0].nooffeatures)[1:-2]), float(str(halloffame[0].mser2_train)[1:-2]), float(str(halloffame[0].mser2_test)[1:-2]), str(halloffame[0])])

    elif poplnType == 'Features' and FtEvlType == 'MSE':
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness),
                       halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), str(halloffame[0])])
    # ------------------------------------------------------------------------- 
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++ Select for Replacement Function +++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
            # for l in aspirants: print(str(l.fitness))
        return chosen

    #+++++++++++++++++++++++++++++++++++++++++++++
    #Breeding Function
    #+++++++++++++++++++++++++++++++++++++++++++++
    # define a breed function as nested.
    def breed():
        # print('breed--------------------')
        nonlocal gen, population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, poolsize, poplnType, FtEvlType, mettarget
        
        #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))
        # p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2, fitness_size=3, parsimony_size=1.4, fitness_first=False)))

        #++++++++ Crossover +++++++++++++++++++++++++++++++++++++
        if random.random() < cxpb:
            p1, p2 = toolbox.mate(p1, p2)
            del p1.fitness.values
            # print('mate done')
        #++++++++ mutation on the offspring ++++++++++++++++               
        if random.random() < mutpb:
            p1, = toolbox.mutate(p1)
            del p1.fitness.values
            # print('mutate done')
            h
        # Evaluate the offspring if it has changed            
        if not p1.fitness.valid:
            # print('not valid after breed operation')
            #++++++++ Counting evaluations +++++++++++++++++
#             counteval_lock.acquire()
#             counteval += 1 #Count the actual evaluations
#             # # NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW 
# #            if counteval % poplnsize == 0:
# #                 print(f'{counteval} evaluations initiated -- {round((100*counteval)/(ngen*poplnsize),2)}% of run {run}')
#             counteval_lock.release()
#             # ``````````````````````````````````````````````````````````````````````````````````````````````````````````

            if poplnType == 'Features' and FtEvlType == 'AdjR2':
                # if not p1.fitness.valid:
                xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(p1, datatrain, datatest)
                p1.evlntime = yo,
                p1.testfitness = zo,
                p1.fitness.values = xo,
                p1.nooffeatures = noft, 
                p1.mser2_train = MSER2_train, 
                p1.mser2_test  = MSER2_test,   
#                print(f'evln {counteval} done - {p1.mser2_train}')
#                counteval = counteval + 1
#                print(f'{counteval}')

                if p1.fitness.values == (0.0101010101010101,) or p1.testfitness == (0.0101010101010101,):
                    p1.fitness.values = 0.0, #for maximising
                    p1.testfitness = 0.0, #for maximising     
                checkTarget = p1.mser2_train[0]
                    # ---------------------------------------------------------


            elif poplnType == 'Features' and FtEvlType == 'MSE':
                xo, yo, zo, noft = toolbox.evaluate(p1, datatrain, datatest)
                p1.evlntime = yo,
                p1.testfitness = zo,
                p1.fitness.values = xo,
                p1.nooffeatures = noft,         

                if p1.fitness.values == (0.0101010101010101,) or p1.testfitness == (0.0101010101010101,):
                    p1.fitness.values = 0.0, #for maximising
                    p1.testfitness = 0.0, #for maximising   
                checkTarget = p1.fitness.values[0]

			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
			#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
#            if float(p1.fitness.values[0]) >= target:
            if checkTarget >= target:
                if mettarget == 0:
                    counteval_lock.acquire()
                    mettarget = counteval
                    counteval_lock.release()
                    print(f'Target met: {counteval}')
                    print(f'Training Fitness: {float(p1.fitness.values[0])}')
#                    targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.fitness.values[0]), 'Met_at': mettarget}, index = {run})
                    targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(checkTarget), 'Met_at': mettarget}, index = {run})
                
                    target_csv = f'{report_csv[:-4]}_Target.csv'
                    #Export from dataframe to CSV file. Update if exists
                    if os.path.isfile(target_csv):
                        targetmet_df.to_csv(target_csv, mode='a', header=False)
                    else:
                        targetmet_df.to_csv(target_csv)                    
			#MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END
                        

            #+++++++++++++++++++++++++++++++++++++++++++++
            # Identify an individual to be replaced - worst fitness
            #+++++++++++++++++++++++++++++++++++++++++++++
                                
            update_lock.acquire()          # LOCK !!!  
            # print('update attempt')
            # Identify a individual to replace from the population. Use Inverse Tournament
            candidates = selInverseTournament(population, k=1, tournsize=5)
            candidate = candidates[0]
            # Replace if offspring is better than candidate individual 
            # updatehof = 'n'# marker to indicate a need update Hall Of Fame
            if p1.fitness.values[0] > candidate.fitness.values[0]: # Max
                # updatehof = 'y' # If replacement done then set marker to update hof
            # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                population.append(p1) 
                population.remove(candidate)
#                 print('replaced')
            
            update_lock.release()            # RELEASE !!!
            #+++++++++++++++++++++++++++++++++++++++++++++
            
            
    # #    Update hall of fame   ????==== INDENT TO HAPPEN ONLY IF A RELPLACEMENT IS DONE ====?????                                                                     
    #         if updatehof == 'y':
    #             try:
    #                 halloffame.update(population)
    #             except AttributeError:
    #                 pass  
    ###########################################################################  


            # ========== After evaluating the individual ======================
            # 00000000000000000000000000000000000000000000000000000000000000000                                
            # ++++++++++++++++++++++++ Counting evaluations +++++++++++++++++++

            # print(f'Pre-update Print >>>>>>>>>>>>>>>>>>>>>>>>, {counteval}')
            counteval_lock.acquire()

            counteval += 1 #Count the actual evaluations
#            print(f'Post update >>>>>>>>>>>>>>>>>>>>>>>>, {counteval}')            #++++++++ Counting evaluations +++++++++++++++++
            if counteval % poplnsize == 0:
                genT = counteval/popsize
                # print('generation -----------------------------------', genT)

                # `````````````````````````````````````````````````````````````
                try:
                    halloffame.update(population)
                except AttributeError:
                    pass  
                # `````````````````````````````````````````````````````````````

                # Lock Population before taking generational stats ````````

                collectStatsGen(genT)	
                # collectStatsGen(genT)					
                if verbose:
                    print(logbook.stream)  

            counteval_lock.release()
            #000000000000000000000000000000000000000000000000000000000000000000
            #==================================================================


    #++++++++++++++++++++++++++++++++++++++++++++++++
    #  GENERATIONAL STATs Collect Depending on Method
    #++++++++++++++++++++++++++++++++++++++++++++++++
    # ----------Crossover - Features OR StdTree ----------------------------------FFFFFFFFF
    if poplnType == 'Features' and FtEvlType == 'AdjR2':
    # ----------------------------------------------------------------------------FFFFFFFFF            
        def collectStatsGen(gen):
            nonlocal population, stats, run, counteval, logbook, verbose, hof_db, halloffame, report_csv
        # def collectStatsGen():
            # nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)

            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]),
						   float(str(halloffame[0].mser2_train)[1:-2]), float(str(halloffame[0].mser2_test)[1:-2]), str(halloffame[0])])

    if poplnType == 'Features' and FtEvlType == 'MSE':
    # ----------------------------------------------------------------------------FFFFFFFFF            
        def collectStatsGen(gen):
            nonlocal population, stats, run, counteval, logbook, verbose, hof_db, halloffame, report_csv
        # def collectStatsGen():
            # nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)

            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), str(halloffame[0])])
    # -----------------------------------------------------------------------------     
    elif poplnType == 'StdTree':
    # ----------------------------------------------------------------------------FFFFFFFFF             
        def collectStatsGen():
            nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)

            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
            #+++++++ END OF GENERATION +++++++++++++++++++

    #++++++++++++++++++++++++++++++++++++++++++++++++++
    #  STATs for a GP Run - Collect Depending on Method
    #++++++++++++++++++++++++++++++++++++++++++++++++++  
    # ----------Crossover - Features OR StdTree ----------------------------------FFFFFFFFF
    if poplnType == 'Features'  and FtEvlType == 'AdjR2':
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
            data = np.array([[*a, *b, *c, *d, *e, *f, *g] for a, b, c, d, e, f, g in zip(*data)])

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
            #       Save 'Hall Of Fame' database         #
            #+++++++++++++++++++++++++++++++++++++++++++++

            # ---- Re-evaluate and Update Evaluation Time ++++++++++++++++++++++++ Features AdjR2 +++++++++++
            for j in range(len(hof_db)):
                # -------------------------------------------------------------
                ind=gp.PrimitiveTree.from_string(hof_db[j][9],pset)
                # -------------------------------------------------------------
                xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(ind, datatrain, datatest)
                hof_db[j][4]=yo
            # -----------------------------------------------------------------+++++++++++++++++++++++++++++++



            #List to dataframe
            hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'No_of_Features', 'MSEr2_train', 'MSEr2_test', 'Best'])
            hof_csv = f'{report_csv[:-4]}_hof.csv' # -------------------------- Destination file (local)
            #Export from dataframe to CSV file. Update if exists
            if os.path.isfile(hof_csv):
                hof_dframe.to_csv(hof_csv, mode='a', header=False)
            else:
                hof_dframe.to_csv(hof_csv)

    # ----------Crossover - Features OR StdTree ----------------------------------FFFFFFFFF
    if poplnType == 'Features'  and FtEvlType == 'MSE':
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
            data = np.array([[*a, *b, *c, *d, *e] for a, b, c, d, e in zip(*data)])
            
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





            # ---- Re-evaluate and Update Evaluation Time +++++++++++++++++++++++++++++++++++Features MSE
            for j in range(len(hof_db)):
                # print(f'old: {hof_db[j][4]}')
                ind=gp.PrimitiveTree.from_string(hof_db[j][7],pset)
                # evalArtificialAnt(ind)
                # hof_db[j][5]
                xo, yo, zo, noft  = toolbox.evaluate(ind, datatrain, datatest)
                hof_db[j][4]=yo
                # print(yo)
                # print(f'new: {hof_db[j][4]}')  
            # -------------------------------------------++++++++++++++++++++++++++++++++++++
            




                
            #List to dataframe
            hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'No_of_Features','Best'])
            hof_csv = f'{report_csv[:-4]}_hof.csv'#Destination file (local)
            #Export from dataframe to CSV file. Update if exists
            if os.path.isfile(hof_csv):
                hof_dframe.to_csv(hof_csv, mode='a', header=False)
            else:
                hof_dframe.to_csv(hof_csv)      

    elif poplnType == 'StdTree':
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
            data = np.array([[*a, *b, *c, *d] for a, b, c, d in zip(*data)])
            
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
            #+++++++++++++++++++++++++++++++++++++++++++++
            
            
            
            
            
            # ---- Re-evaluate and Update Evaluation Time +++++++++++++++++++++++++++++++++++ stdtree 
            for j in range(len(hof_db)):
                # print(f'old: {hof_db[j][4]}')
                ind=gp.PrimitiveTree.from_string(hof_db[j][6],pset)
                # evalArtificialAnt(ind)
                # hof_db[j][5]
                xo, yo, uo = toolbox.evaluate(ind, datatrain, datatest)
                hof_db[j][4]=yo
                # print(yo)
                # print(f'new: {hof_db[j][4]}')  
            # -------------------------------------------++++++++++++++++++++++++++++++++++++
                        
            
            
            
             
            
            
            
            #List to dataframe
            hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
            hof_csv = f'{report_csv[:-4]}_hof.csv'#Destination file (local)
            #Export from dataframe to CSV file. Update if exists
            if os.path.isfile(hof_csv):
                hof_dframe.to_csv(hof_csv, mode='a', header=False)
            else:
                hof_dframe.to_csv(hof_csv)

##++++++++++++++++++++++++++++++++++++++++++++++++++
##  STATs Crossover Effect
##++++++++++++++++++++++++++++++++++++++++++++++++++  
#    def collectcxOverStats():
#        nonlocal run, gen, B4Fitness, B4Test_Fitness, AfFitness, AfTest_Fitness, report_csv, cxOver_db     
#    #+++++++++++++++++++++++++++++++++++++++++++++
#    ## Save Crossover Stats
#    #++++++++++++++++++++++++++++++++++++++++++++++
#        #List to dataframe
#        cxOver_dframe=pd.DataFrame(cxOver_db, columns=['Run', 'Generation', 'Train_Fitness_imp', 'Test_Fitness_imp'])
##        cxOver_dframe=pd.DataFrame(cxOver_db, columns=['Run', 'Generation', 'B4Fitness', 'B4Test_Fitness', 'AfFitness', 'AfTest_Fitness'])
#        cxOver_csv = f'{report_csv[:-4]}_cxOver.csv'#Destination file (local)
#        #Export from dataframe to CSV file. Update if exists
#        if os.path.isfile(cxOver_csv):
#            cxOver_dframe.to_csv(cxOver_csv, mode='a', header=False)
#        else:
#            cxOver_dframe.to_csv(cxOver_csv)
#        cxOver_db=[]

# # NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW 
# ##+++++++++++++++++++++++++++++++++++++++++++++
# ##Create a Generation
# ##+++++++++++++++++++++++++++++++++++++++++++++
# #    # Begin the generational process   
# #    for gen in range(1, ngen+1):
#     tp = ThreadPool(poolsize)  # <-------------------------------------------- (3a)
#         # Generate offsprings -  equivalent to a generation based on populations size
#     poplnsize =  len(population)
#     targetevalns = poplnsize*ngen
#     counteval = 0 
# #        while (counteval < poplnsize+1):  
#     for h in range(int(poplnsize*ngen*factor)):
# #        print(counteval)
#         tp.apply_async(breed)

# #   Append the current generation statistics to the logbook
#     tp.close() # <---------------------------------------??????????????
#     tp.join() #  <---------------------------------------??????????????
#     print(f'done  : {counteval}')
#     print(f'Target: {targetevalns}')
#     print(threading.active_count())
# #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''	<--------------------????????????????????????????
    
#    # Begin the generational process   
    tp = ThreadPool(poolsize)  # <---------------------------------------------
    # Generate offsprings -  equivalent to a generation based on populations size
    poplnsize =  len(population)
    targetevalns = poplnsize*ngen
    
    # counteval = 0 
    for h in range(int(poplnsize*ngen*factor)):# no. of breed operation attempt determined by probabilities
        tp.apply_async(breed)

#   Append the current generation statistics to the logbook
    tp.close() # <--------------------------------------- ??????????????
    tp.join() #  <--------------------------------------- ??????????????
    print('<--------------------------------------- ??????????????')
    # If last generation is not complete do a few more breed operations
    while counteval < targetevalns:
        #print(f'more pending')
        tp = ThreadPool(poolsize)  # <-----------------------------------------
        #   if count < target:psize
        for j in range(targetevalns - counteval):
            tp.apply_async(breed)
        tp.close() # <---------------------------------------  ??????????????
        tp.join() #  <---------------------------------------  ??????????????

    print(f'done  : {counteval}')
    print(f'Target: {targetevalns}')
    print(threading.active_count())

	
    # collectStatsGen()
    collectStatsRun()    
    
    
    
    
# #+++++++++++++++++++++++++++++++++++++++++++++
#     for ind in population:
#         if poplnType == 'Features' and FtEvlType == 'AdjR2':
# #            if not ind.fitness.valid:
#             xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(ind, datatrain, datatest)
#             ind.evlntime = yo,
#             ind.testfitness = zo,
#             ind.fitness.values = xo,
#             ind.nooffeatures = noft, 
#             ind.mser2_train = MSER2_train, 
#             ind.mser2_test  = MSER2_test,             

#             if ind.fitness.values == (0.0101010101010101,) :
#                 ind.fitness.values = 0.0, #for maximising
#             if ind.testfitness == (0.0101010101010101,) :
#                 ind.testfitness = 0.0, #for maximising         

#         elif poplnType == 'Features' and FtEvlType == 'MSE':
#             xo, yo, zo, noft = toolbox.evaluate(ind, datatrain, datatest)
#             ind.evlntime = yo,
#             ind.testfitness = zo,
#             ind.fitness.values = xo,
#             ind.nooffeatures = noft,         

#             if ind.fitness.values == (0.0101010101010101,) :
#                 ind.fitness.values = 0.0, #for maximising
#             if ind.testfitness == (0.0101010101010101,) :
#                 ind.testfitness = 0.0, #for maximising   
#     #+++++++++++++++++++++++++++++++++++++++++++++
#     # collectStatsGen()
#     collectStatsRun()
# # `````````````````````````````````````````````````````````````````````````````   
   
    
###############################################################################       
    return population, logbook    
###############################################################################
#


"""
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           FEATURES: GP - Steady State - Crossover Stats (NMSE or AdjR2)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
def gpDoubleCx(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, datatrain=None, datatest=None, target =None, poplnType =None,FtEvlType =None):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
    counteval = 0
    countdiverr = 0
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    mettarget = 0 # 0 = not set
	#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END\    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    update_lock = threading.Lock()

    # -------------------------------------------------------------------------
    logbook = tools.Logbook()
    logbook.header = ['run', 'gen', 'nevals'] + (stats.fields if stats else [])
    
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Evaluation of Initial Population  (NMSE or AdjR2)
    #+++++++++++++++++++++++++++++++++++++++++++++
    for ind in population:
        if poplnType == 'Features' and FtEvlType == 'AdjR2':
            if not ind.fitness.valid:
                xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(ind, datatrain, datatest)
                ind.evlntime = yo,
                ind.testfitness = zo,
                ind.fitness.values = xo,
                ind.nooffeatures = noft, 
                ind.mser2_train = MSER2_train, 
                ind.mser2_test  = MSER2_test,             
                if ind.fitness.values == (0.0101010101010101,) or ind.testfitness == (0.0101010101010101,):
                    ind.fitness.values = 0.0, #for maximising
                    ind.testfitness = 0.0, #for maximising     
                    countdiverr += 1

        elif poplnType == 'Features' and FtEvlType == 'MSE':
                xo, yo, zo, noft = toolbox.evaluate(ind, datatrain, datatest)
                ind.evlntime = yo,
                ind.testfitness = zo,
                ind.fitness.values = xo,
                ind.nooffeatures = noft,         
                if ind.fitness.values == (0.0101010101010101,) or ind.testfitness == (0.0101010101010101,):
                    ind.fitness.values = 0.0, #for maximising
                    ind.testfitness = 0.0, #for maximising     
                    countdiverr += 1
    #+++++++++++++++++++++++++++++++++++++++++++++
    try:
        halloffame.update(population)
    except AttributeError:
        pass
    #+++++++++++++++++++++++++++++++++++++++++++++
    record = stats.compile(population) if stats else {}
    logbook.record(run=run, gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)  
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Capture best individual for Generation 0
    gen=0
    hof_db=[]
    cxOver_db=[] # -------------------77777777777777777777777777777777777777777
    B4Fitness = 0
    B4Test_Fitness = 0
    AfFitness = 0
    AfTest_Fitness = 0
    # -------------------------------------------------------------------------
    # Collect HOF Data  (NMSE or AdjR2) 
    if poplnType == 'Features' and FtEvlType == 'AdjR2':
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), halloffame[0].evlntime, len(halloffame[0]),
                       int(str(halloffame[0].nooffeatures)[1:-2]), float(str(halloffame[0].mser2_train)[1:-2]), float(str(halloffame[0].mser2_test)[1:-2]), str(halloffame[0])])

    elif poplnType == 'Features' and FtEvlType == 'MSE':
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness),
                       halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), str(halloffame[0])])
    # ------------------------------------------------------------------------- 
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++ Select for Replacement Function +++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    # Breeding Function - TWO OFFSPRINGS
    #+++++++++++++++++++++++++++++++++++++++++++++
    # define a breed function.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, cxOver_db, mettarget, countdiverr

        # initialise ----------------777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777
#        B4Fitness = 0
#        B4Test_Fitness = 0
#        AfFitness = 0
#        AfTest_Fitness = 0
        offspring=[]
        successCX = False
        #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2, fitness_size=3, parsimony_size=1.4, fitness_first=False)))
        # p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))
        # Fitness Before CrossOver -------------7777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777
#        B4Fitness1 = p1.fitness.values[0]
#        B4Test_Fitness1 = p1.testfitness[0]
#        
#        B4Fitness2 = p2.fitness.values[0]
#        B4Test_Fitness2 = p2.testfitness[0]
        #++++++++ Crossover +++++++++++++++++++++++++++++++++++++
        if random.random() < cxpb:
            p1, p2 = toolbox.mate(p1, p2)
            del p1.fitness.values
            del p2.fitness.values
#            successCX = True
        #   TWO OFFSPRING  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        offspring = p1,# p2
#        count=0
        for cand in offspring:  
#            str(cand)
#            count += 1
            #++++++++ mutation on the offspring ++++++++++++++++               
            if random.random() < mutpb:
                cand, = toolbox.mutate(cand)
                del cand.fitness.values
#                print(f'mutated {cand}')
            # Evaluate the offspring if it has changed
            # @@@@@@@@@@@@@@@@@(NMSE or AdjR2)@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if poplnType == 'Features' and FtEvlType == 'MSE':
                if not cand.fitness.valid:
                    #++++++++ Counting evaluations +++++++++++++++++
#                    counteval_lock.acquire()
                    counteval += 1 #Count the actual evaluations
#                    counteval_lock.release()
                    xo, yo, zo, noft = toolbox.evaluate(cand, datatrain, datatest)
                    cand.evlntime = yo,
                    cand.testfitness = zo,
                    cand.fitness.values = xo, 
                    cand.nooffeatures = noft,
                    # Check if ZeroDivisionError, ValueError 
#                    if cand.fitness.values == (0.0101010101010101,) :
#                        cand.fitness.values = 0.0, #for maximising
#                    if cand.testfitness == (0.0101010101010101,) :
#                        cand.testfitness = 0.0, #for maximising  

                    if cand.fitness.values == (0.0101010101010101,) or cand.testfitness == (0.0101010101010101,):
                        cand.fitness.values = 0.0, #for maximising
                        cand.testfitness = 0.0, #for maximising     
                        countdiverr += 1

                if float(cand.fitness.values[0]) >= target: # ---------------------------- TRAINING MSE
                    if mettarget == 0:
                        mettarget = counteval
                        print(f'Target met: {counteval}')
                        print(f'Training Fitness: {float(cand.fitness.values[0])}')
                        targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(cand.fitness.values[0]), 'Met_at': mettarget}, index = {run})
                    
                        target_csv = f'{report_csv[:-4]}_Target.csv'
                        #Export from dataframe to CSV file. Update if exists
                        if os.path.isfile(target_csv):
                            targetmet_df.to_csv(target_csv, mode='a', header=False)
                        else:
                            targetmet_df.to_csv(target_csv)   
                            
            elif poplnType == 'Features' and FtEvlType == 'AdjR2':
                if not cand.fitness.valid:
                    #++++++++ Counting evaluations +++++++++++++++++
#                    counteval_lock.acquire()
                    counteval += 1 #Count the actual evaluations
#                    counteval_lock.release()
                    xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(cand, datatrain, datatest)
#                    xo, yo, zo = toolbox.evaluate(p1)                
                    cand.evlntime = yo,
                    cand.testfitness = zo,
                    cand.fitness.values = xo, 
                    cand.nooffeatures = noft,
                    cand.mser2_train = MSER2_train, 
                    cand.mser2_test  = MSER2_test,
                    #Check if ZeroDivisionError, ValueError 
#                    if cand.fitness.values == (0.0101010101010101,) :
#                        cand.fitness.values = 0.0, #for maximising
#                    if cand.testfitness == (0.0101010101010101,) :
#                        cand.testfitness = 0.0, #for maximising  
                    if cand.fitness.values == (0.0101010101010101,) or cand.testfitness == (0.0101010101010101,):
                        cand.fitness.values = 0.0, #for maximising
                        cand.testfitness = 0.0, #for maximising     
                        countdiverr += 1                        
             #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@            

    			#[[[[[[[[[[[[[[[[[[[[[[[[[[ TARGET MET ? ]]]]]]]]]]]]]]]]]BEGIN
    			#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
   
                if float(cand.mser2_train[0]) >= target: # ---------------------------- TRAINING MSE
                    if mettarget == 0:
                        mettarget = counteval
                        print(f'Target met: {counteval}')
                        print(f'Training Fitness: {float(cand.fitness.values[0])}')
                        targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(cand.mser2_train[0]), 'Met_at': mettarget}, index = {run})
                    
                        target_csv = f'{report_csv[:-4]}_Target.csv'
                        #Export from dataframe to CSV file. Update if exists
                        if os.path.isfile(target_csv):
                            targetmet_df.to_csv(target_csv, mode='a', header=False)
                        else:
                            targetmet_df.to_csv(target_csv)  
    			#MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]END

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # REPLACEMENT - worst fitness from random k
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#            update_lock.acquire()          # LOCK !!!  
            # Identify a individual to replace from the population. Use Inverse Tournament
            candidates = selInverseTournament(population, k=1, tournsize=5)
            candidate = candidates[0]
            # Replace if offspring is better than candidate individual 
            if cand.fitness.values[0] > candidate.fitness.values[0]: # Max
            # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                    population.append(cand) 
                    population.remove(candidate)
#                    print(f'{count} replaced')
#            update_lock.release()            # RELEASE !!!
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Update hall of fame ---------------------------------------------                                                                    
            try:
                halloffame.update(population)
            except AttributeError:
                pass       
            # Record Fitness Values after Cross-Over
#            if successCX == True:
#                if count == 1:
#                    B4Fitness = B4Fitness1
#                    B4Test_Fitness = B4Test_Fitness1
#                else:
#                    B4Fitness = B4Fitness2
#                    B4Test_Fitness = B4Test_Fitness2
#                    
#                AfFitness = cand.fitness.values[0]
#                AfTest_Fitness = cand.testfitness[0]
#                
#                try:
#                    Train_imp = round(100*(AfFitness - B4Fitness)/B4Fitness)
#                except ZeroDivisionError:
#                    Train_imp = 0
#                
#                try:
#                    Test_imp  = round(100*(AfTest_Fitness - B4Test_Fitness)/B4Test_Fitness)
#                except ZeroDivisionError:
#                    Test_imp = 0
#                    
#                cxOver_db.append([run, gen, Train_imp, Test_imp])

#++++++++++++++++++++++++++++++++++++++++++++++++
#  GENERATIONAL STATs Collect Depending on Method
#++++++++++++++++++++++++++++++++++++++++++++++++
    # ----------Crossover - Features OR StdTree ----------------------------------FFFFFFFFF
    if poplnType == 'Features' and FtEvlType == 'AdjR2':
    # ----------------------------------------------------------------------------FFFFFFFFF            
        def collectStatsGen():
            nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)
            
            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), float(str(halloffame[0].mser2_train)[1:-2]), float(str(halloffame[0].mser2_test)[1:-2]), str(halloffame[0])])

    if poplnType == 'Features' and FtEvlType == 'MSE':
    # ----------------------------------------------------------------------------FFFFFFFFF            
        def collectStatsGen():
            nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)
            
            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), str(halloffame[0])])
    # -----------------------------------------------------------------------------     
    elif poplnType == 'StdTree':
    # ----------------------------------------------------------------------------FFFFFFFFF             
        def collectStatsGen():
            nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)
            
            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
            #+++++++ END OF GENERATION +++++++++++++++++++
            
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    #  STATs for a GP Run - Collect Depending on Method
    #++++++++++++++++++++++++++++++++++++++++++++++++++  
    # ----------Crossover - Features OR StdTree ----------------------------------FFFFFFFFF
    if poplnType == 'Features'  and FtEvlType == 'AdjR2':
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
            data = np.array([[*a, *b, *c, *d, *e, *f, *g] for a, b, c, d, e, f, g in zip(*data)])
            
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
            hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'No_of_Features', 'MSEr2_train', 'MSEr2_test', 'Best'])
            hof_csv = f'{report_csv[:-4]}_hof.csv'#Destination file (local)
            #Export from dataframe to CSV file. Update if exists
            if os.path.isfile(hof_csv):
                hof_dframe.to_csv(hof_csv, mode='a', header=False)
            else:
                hof_dframe.to_csv(hof_csv)
    
    # ----------Crossover - Features OR StdTree ----------------------------------FFFFFFFFF
    if poplnType == 'Features'  and FtEvlType == 'MSE':
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
            data = np.array([[*a, *b, *c, *d, *e] for a, b, c, d, e in zip(*data)])
            
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
            hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'No_of_Features','Best'])
            hof_csv = f'{report_csv[:-4]}_hof.csv'#Destination file (local)
            #Export from dataframe to CSV file. Update if exists
            if os.path.isfile(hof_csv):
                hof_dframe.to_csv(hof_csv, mode='a', header=False)
            else:
                hof_dframe.to_csv(hof_csv)      

    elif poplnType == 'StdTree':
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
            data = np.array([[*a, *b, *c, *d] for a, b, c, d in zip(*data)])
            
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
            #+++++++++++++++++++++++++++++++++++++++++++++
            #List to dataframe
            hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
            hof_csv = f'{report_csv[:-4]}_hof.csv'#Destination file (local)
            #Export from dataframe to CSV file. Update if exists
            if os.path.isfile(hof_csv):
                hof_dframe.to_csv(hof_csv, mode='a', header=False)
            else:
                hof_dframe.to_csv(hof_csv)

    #++++++++++++++++++++++++++++++++++++++++++++++++++
    #  STATs Crossover Effect
    #++++++++++++++++++++++++++++++++++++++++++++++++++  
    def collectcxOverStats():
        nonlocal run, gen, B4Fitness, B4Test_Fitness, AfFitness, AfTest_Fitness, report_csv, cxOver_db     
    #+++++++++++++++++++++++++++++++++++++++++++++
    ## Save Crossover Stats
    #++++++++++++++++++++++++++++++++++++++++++++++
        #List to dataframe
        cxOver_dframe=pd.DataFrame(cxOver_db, columns=['Run', 'Generation', 'Train_Fitness_imp', 'Test_Fitness_imp'])
#        cxOver_dframe=pd.DataFrame(cxOver_db, columns=['Run', 'Generation', 'B4Fitness', 'B4Test_Fitness', 'AfFitness', 'AfTest_Fitness'])
        cxOver_csv = f'{report_csv[:-4]}_cxOver.csv'#Destination file (local)
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(cxOver_csv):
            cxOver_dframe.to_csv(cxOver_csv, mode='a', header=False)
        else:
            cxOver_dframe.to_csv(cxOver_csv)
        cxOver_db=[]
		
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Create a Generation
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Begin the generational process
    for gen in range(1, ngen+1):
        # Generate offsprings -  equivalent to a generation / populations size
        poplnsize =  len(population)
#        poplnsize =  500
        # print(f'Number of div errors in generation {gen} = {countdiverr}')
        counteval = 0 
        countdiverr = 0
#        for h in range(poplnsize):
#            breed()
        while counteval < poplnsize:
            breed()
#            for j in range(poplnsize - counteval):
#                breed()
#        collectcxOverStats() #---------------------------------------------------FFFFFFFFF
        collectStatsGen()
    collectStatsRun()
    ###############################################################################       
    return population, logbook    
    ###############################################################################


def gpFTStd(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, datatrain=None, datatest=None, target =None, poplnType =None,FtEvlType =None):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
    counteval = 0
    countdiverr = 0
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    mettarget = 0 # 0 = not set
	#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END
	#``````````````````````````````````````````````````````````````````````````````  
    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    # -------------------------------------------------------------------------
    logbook = tools.Logbook()
    logbook.header = ['run', 'gen', 'nevals'] + (stats.fields if stats else [])
    
    
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Evaluation of Initial Population  (NMSE or AdjR2)
    #+++++++++++++++++++++++++++++++++++++++++++++
    for ind in population:
        if poplnType == 'Features' and FtEvlType == 'AdjR2':
            if not ind.fitness.valid:
                xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(ind, datatrain, datatest)
                ind.evlntime = yo,
                ind.testfitness = zo,
                ind.fitness.values = xo,
                ind.nooffeatures = noft, 
                ind.mser2_train = MSER2_train, 
                ind.mser2_test  = MSER2_test,             

#                if ind.fitness.values == (0.0101010101010101,) :
#                    ind.fitness.values = 0.0, #for maximising
#                if ind.testfitness == (0.0101010101010101,) :
#                    ind.testfitness = 0.0, #for maximising     
#                    countdiverr += 1
                if ind.fitness.values == (0.0101010101010101,) or ind.testfitness == (0.0101010101010101,):
                    ind.fitness.values = 0.0, #for maximising
                    ind.testfitness = 0.0, #for maximising     
                    countdiverr += 1

        elif poplnType == 'Features' and FtEvlType == 'MSE':
                xo, yo, zo, noft = toolbox.evaluate(ind, datatrain, datatest)
                ind.evlntime = yo,
                ind.testfitness = zo,
                ind.fitness.values = xo,
                ind.nooffeatures = noft,         

#                if ind.fitness.values == (0.0101010101010101,) :
#                    ind.fitness.values = 0.0, #for maximising
#                if ind.testfitness == (0.0101010101010101,) :
#                    ind.testfitness = 0.0, #for maximising   
                if ind.fitness.values == (0.0101010101010101,) or ind.testfitness == (0.0101010101010101,):
                    ind.fitness.values = 0.0, #for maximising
                    ind.testfitness = 0.0, #for maximising     
                    countdiverr += 1
                    
    #+++++++++++++++++++++++++++++++++++++++++++++
    try:
        halloffame.update(population)
    except AttributeError:
        pass
    #+++++++++++++++++++++++++++++++++++++++++++++
    record = stats.compile(population) if stats else {}
    logbook.record(run=run, gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)  
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Capture best individual for Generation 0
    gen=0
    hof_db=[]
    cxOver_db=[] # -------------------77777777777777777777777777777777777777777
    B4Fitness = 0
    B4Test_Fitness = 0
    AfFitness = 0
    AfTest_Fitness = 0
    # -------------------------------------------------------------------------
    # Collect HOF Data  (NMSE or AdjR2) 
    if poplnType == 'Features' and FtEvlType == 'AdjR2':
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), halloffame[0].evlntime, len(halloffame[0]),
                       int(str(halloffame[0].nooffeatures)[1:-2]), float(str(halloffame[0].mser2_train)[1:-2]), float(str(halloffame[0].mser2_test)[1:-2]), str(halloffame[0])])

    elif poplnType == 'Features' and FtEvlType == 'MSE':
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness),
                       halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), str(halloffame[0])])
    # ------------------------------------------------------------------------- 
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++ Select for Replacement Function +++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    # Breeding Function - TWO OFFSPRINGS
    #+++++++++++++++++++++++++++++++++++++++++++++
    # define a breed function.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, cxOver_db, mettarget, countdiverr

        # initialise ----------------777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777
#        B4Fitness = 0
#        B4Test_Fitness = 0
#        AfFitness = 0
#        AfTest_Fitness = 0
        offspring=[]
        successCX = False
        #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        # print('selection attempt -------')
        # p1, p2 = list(map(toolbox.clone, tctools.selDoubleTournTime(population, 2, fitness_size=3, parsimony_size=1.4, fitness_first=False)))
        # selDoubleTournTime(individuals, k, fitness_size, parsimony_size, fitness_first, fit_attr="fitness")
        # p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))
        # print(str(p1))
        # Fitness Before CrossOver -------------7777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777
#        B4Fitness1 = p1.fitness.values[0]
#        B4Test_Fitness1 = p1.testfitness[0]
#        
#        B4Fitness2 = p2.fitness.values[0]
#        B4Test_Fitness2 = p2.testfitness[0]
        #++++++++ Crossover +++++++++++++++++++++++++++++++++++++
        if random.random() < cxpb:
            p1, p2 = toolbox.mate(p1, p2)
            del p1.fitness.values
            del p2.fitness.values
#            successCX = True
        #   TWO OFFSPRING  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        offspring = p1,# p2
#        count=0
        for cand in offspring:  
#            str(cand)
#            count += 1
            #++++++++ mutation on the offspring ++++++++++++++++               
            if random.random() < mutpb:
                cand, = toolbox.mutate(cand)
                del cand.fitness.values
#                print(f'mutated {cand}')
            # Evaluate the offspring if it has changed
            # @@@@@@@@@@@@@@@@@(NMSE or AdjR2)@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if poplnType == 'Features' and FtEvlType == 'MSE':
                if not cand.fitness.valid:
                    #++++++++ Counting evaluations +++++++++++++++++
#                    counteval_lock.acquire()
                    counteval += 1 #Count the actual evaluations
#                    counteval_lock.release()
                    xo, yo, zo, noft = toolbox.evaluate(cand, datatrain, datatest)
                    cand.evlntime = yo,
                    cand.testfitness = zo,
                    cand.fitness.values = xo, 
                    cand.nooffeatures = noft,
                    # Check if ZeroDivisionError, ValueError 
#                    if cand.fitness.values == (0.0101010101010101,) :
#                        cand.fitness.values = 0.0, #for maximising
#                    if cand.testfitness == (0.0101010101010101,) :
#                        cand.testfitness = 0.0, #for maximising  

                    if cand.fitness.values == (0.0101010101010101,) or cand.testfitness == (0.0101010101010101,):
                        cand.fitness.values = 0.0, #for maximising
                        cand.testfitness = 0.0, #for maximising     
                        countdiverr += 1

                if float(cand.fitness.values[0]) >= target: # ---------------------------- TRAINING MSE
                    if mettarget == 0:
                        mettarget = counteval
                        print(f'Target met: {counteval}')
                        print(f'Training Fitness: {float(cand.fitness.values[0])}')
                        targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(cand.fitness.values[0]), 'Met_at': mettarget}, index = {run})
                    
                        target_csv = f'{report_csv[:-4]}_Target.csv'
                        #Export from dataframe to CSV file. Update if exists
                        if os.path.isfile(target_csv):
                            targetmet_df.to_csv(target_csv, mode='a', header=False)
                        else:
                            targetmet_df.to_csv(target_csv)   
                            
            elif poplnType == 'Features' and FtEvlType == 'AdjR2':
                if not cand.fitness.valid:
                    #++++++++ Counting evaluations +++++++++++++++++
#                    counteval_lock.acquire()
                    counteval += 1 #Count the actual evaluations
#                    counteval_lock.release()
                    xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(cand, datatrain, datatest)
#                    xo, yo, zo = toolbox.evaluate(p1)                
                    cand.evlntime = yo,
                    cand.testfitness = zo,
                    cand.fitness.values = xo, 
                    cand.nooffeatures = noft,
                    cand.mser2_train = MSER2_train, 
                    cand.mser2_test  = MSER2_test,
                    #Check if ZeroDivisionError, ValueError 
#                    if cand.fitness.values == (0.0101010101010101,) :
#                        cand.fitness.values = 0.0, #for maximising
#                    if cand.testfitness == (0.0101010101010101,) :
#                        cand.testfitness = 0.0, #for maximising  
                    if cand.fitness.values == (0.0101010101010101,) or cand.testfitness == (0.0101010101010101,):
                        cand.fitness.values = 0.0, #for maximising
                        cand.testfitness = 0.0, #for maximising     
                        countdiverr += 1                        
             #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@            

    			#[[[[[[[[[[[[[[[[[[[[[[[[[[ TARGET MET ? ]]]]]]]]]]]]]]]]]BEGIN
    			#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
   
                if float(cand.mser2_train[0]) >= target: # ---------------------------- TRAINING MSE
                    if mettarget == 0:
                        mettarget = counteval
                        print(f'Target met: {counteval}')
                        print(f'Training Fitness: {float(cand.fitness.values[0])}')
                        targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(cand.mser2_train[0]), 'Met_at': mettarget}, index = {run})
                    
                        target_csv = f'{report_csv[:-4]}_Target.csv'
                        #Export from dataframe to CSV file. Update if exists
                        if os.path.isfile(target_csv):
                            targetmet_df.to_csv(target_csv, mode='a', header=False)
                        else:
                            targetmet_df.to_csv(target_csv)  
    			#MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]END

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # REPLACEMENT - worst fitness from random k
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#            update_lock.acquire()          # LOCK !!!  
            # Identify a individual to replace from the population. Use Inverse Tournament
            candidates = selInverseTournament(population, k=1, tournsize=5)
            candidate = candidates[0]
            # Replace if offspring is better than candidate individual 
            if cand.fitness.values[0] > candidate.fitness.values[0]: # Max
            # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                    population.append(cand) 
                    population.remove(candidate)
#                    print(f'{count} replaced')
#            update_lock.release()            # RELEASE !!!
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Update hall of fame ---------------------------------------------                                                                    
            try:
                halloffame.update(population)
            except AttributeError:
                pass       
            # Record Fitness Values after Cross-Over
#            if successCX == True:
#                if count == 1:
#                    B4Fitness = B4Fitness1
#                    B4Test_Fitness = B4Test_Fitness1
#                else:
#                    B4Fitness = B4Fitness2
#                    B4Test_Fitness = B4Test_Fitness2
#                    
#                AfFitness = cand.fitness.values[0]
#                AfTest_Fitness = cand.testfitness[0]
#                
#                try:
#                    Train_imp = round(100*(AfFitness - B4Fitness)/B4Fitness)
#                except ZeroDivisionError:
#                    Train_imp = 0
#                
#                try:
#                    Test_imp  = round(100*(AfTest_Fitness - B4Test_Fitness)/B4Test_Fitness)
#                except ZeroDivisionError:
#                    Test_imp = 0
#                    
#                cxOver_db.append([run, gen, Train_imp, Test_imp])

#++++++++++++++++++++++++++++++++++++++++++++++++
#  GENERATIONAL STATs Collect Depending on Method
#++++++++++++++++++++++++++++++++++++++++++++++++
    # ----------Crossover - Features OR StdTree ----------------------------------FFFFFFFFF
    if poplnType == 'Features' and FtEvlType == 'AdjR2':
    # ----------------------------------------------------------------------------FFFFFFFFF            
        def collectStatsGen():
            nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)
            
            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), float(str(halloffame[0].mser2_train)[1:-2]), float(str(halloffame[0].mser2_test)[1:-2]), str(halloffame[0])])


    if poplnType == 'Features' and FtEvlType == 'MSE':
    # ----------------------------------------------------------------------------FFFFFFFFF            
        def collectStatsGen():
            nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)
            
            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), str(halloffame[0])])

    # -----------------------------------------------------------------------------     
    elif poplnType == 'StdTree':
    # ----------------------------------------------------------------------------FFFFFFFFF             
        def collectStatsGen():
            nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)
            
            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
            #+++++++ END OF GENERATION +++++++++++++++++++
            
#++++++++++++++++++++++++++++++++++++++++++++++++++
#  STATs for a GP Run - Collect Depending on Method
#++++++++++++++++++++++++++++++++++++++++++++++++++  
    # ----------Crossover - Features OR StdTree ----------------------------------FFFFFFFFF
    if poplnType == 'Features'  and FtEvlType == 'AdjR2':
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
            data = np.array([[*a, *b, *c, *d, *e, *f, *g] for a, b, c, d, e, f, g in zip(*data)])
            
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
            hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'No_of_Features', 'MSEr2_train', 'MSEr2_test', 'Best'])
            hof_csv = f'{report_csv[:-4]}_hof.csv'#Destination file (local)
            #Export from dataframe to CSV file. Update if exists
            if os.path.isfile(hof_csv):
                hof_dframe.to_csv(hof_csv, mode='a', header=False)
            else:
                hof_dframe.to_csv(hof_csv)
    
    # ----------Crossover - Features OR StdTree ----------------------------------FFFFFFFFF
    if poplnType == 'Features'  and FtEvlType == 'MSE':
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
            data = np.array([[*a, *b, *c, *d, *e] for a, b, c, d, e in zip(*data)])
            
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
            hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'No_of_Features','Best'])
            hof_csv = f'{report_csv[:-4]}_hof.csv'#Destination file (local)
            #Export from dataframe to CSV file. Update if exists
            if os.path.isfile(hof_csv):
                hof_dframe.to_csv(hof_csv, mode='a', header=False)
            else:
                hof_dframe.to_csv(hof_csv)      

    elif poplnType == 'StdTree':
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
            data = np.array([[*a, *b, *c, *d] for a, b, c, d in zip(*data)])
            
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
            #+++++++++++++++++++++++++++++++++++++++++++++
            #List to dataframe
            hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
            hof_csv = f'{report_csv[:-4]}_hof.csv'#Destination file (local)
            #Export from dataframe to CSV file. Update if exists
            if os.path.isfile(hof_csv):
                hof_dframe.to_csv(hof_csv, mode='a', header=False)
            else:
                hof_dframe.to_csv(hof_csv)

    #++++++++++++++++++++++++++++++++++++++++++++++++++
    #  STATs Crossover Effect
    #++++++++++++++++++++++++++++++++++++++++++++++++++  
    def collectcxOverStats():
        nonlocal run, gen, B4Fitness, B4Test_Fitness, AfFitness, AfTest_Fitness, report_csv, cxOver_db     
    #+++++++++++++++++++++++++++++++++++++++++++++
    ## Save Crossover Stats
    #++++++++++++++++++++++++++++++++++++++++++++++
        #List to dataframe
        cxOver_dframe=pd.DataFrame(cxOver_db, columns=['Run', 'Generation', 'Train_Fitness_imp', 'Test_Fitness_imp'])
#        cxOver_dframe=pd.DataFrame(cxOver_db, columns=['Run', 'Generation', 'B4Fitness', 'B4Test_Fitness', 'AfFitness', 'AfTest_Fitness'])
        cxOver_csv = f'{report_csv[:-4]}_cxOver.csv'#Destination file (local)
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(cxOver_csv):
            cxOver_dframe.to_csv(cxOver_csv, mode='a', header=False)
        else:
            cxOver_dframe.to_csv(cxOver_csv)
        cxOver_db=[]
		
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Create a Generation
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Begin the generational process
    for gen in range(1, ngen+1):
        # Generate offsprings -  equivalent to a generation / populations size
        poplnsize =  len(population)
#        poplnsize =  500
        # print(f'Number of div errors in generation {gen} = {countdiverr}')
        counteval = 0 
        countdiverr = 0
#        for h in range(poplnsize):
#            breed()
        while counteval < poplnsize:
            breed()
#            for j in range(poplnsize - counteval):
#                breed()
#        collectcxOverStats() #---------------------------------------------------FFFFFFFFF
        collectStatsGen()
    collectStatsRun()
    ###############################################################################       
    return population, logbook    
    ###############################################################################



def gpDoubleTC(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, datatrain=None, datatest=None, target =None, poplnType =None,FtEvlType =None):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
    counteval = 0
    countdiverr = 0
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    mettarget = 0 # 0 = not set
	#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END
	#``````````````````````````````````````````````````````````````````````````````  
    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    # -------------------------------------------------------------------------
    logbook = tools.Logbook()
    logbook.header = ['run', 'gen', 'nevals'] + (stats.fields if stats else [])
    
    
    #+++++++++++++++++++++++++++++++++++++++++++++
    #Evaluation of Initial Population  (NMSE or AdjR2)
    #+++++++++++++++++++++++++++++++++++++++++++++
    for ind in population:
        if poplnType == 'Features' and FtEvlType == 'AdjR2':
            if not ind.fitness.valid:
                xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(ind, datatrain, datatest)
                ind.evlntime = yo,
                ind.testfitness = zo,
                ind.fitness.values = xo,
                ind.nooffeatures = noft, 
                ind.mser2_train = MSER2_train, 
                ind.mser2_test  = MSER2_test,             

#                if ind.fitness.values == (0.0101010101010101,) :
#                    ind.fitness.values = 0.0, #for maximising
#                if ind.testfitness == (0.0101010101010101,) :
#                    ind.testfitness = 0.0, #for maximising     
#                    countdiverr += 1
                if ind.fitness.values == (0.0101010101010101,) or ind.testfitness == (0.0101010101010101,):
                    ind.fitness.values = 0.0, #for maximising
                    ind.testfitness = 0.0, #for maximising     
                    countdiverr += 1

        elif poplnType == 'Features' and FtEvlType == 'MSE':
                xo, yo, zo, noft = toolbox.evaluate(ind, datatrain, datatest)
                ind.evlntime = yo,
                ind.testfitness = zo,
                ind.fitness.values = xo,
                ind.nooffeatures = noft,         

#                if ind.fitness.values == (0.0101010101010101,) :
#                    ind.fitness.values = 0.0, #for maximising
#                if ind.testfitness == (0.0101010101010101,) :
#                    ind.testfitness = 0.0, #for maximising   
                if ind.fitness.values == (0.0101010101010101,) or ind.testfitness == (0.0101010101010101,):
                    ind.fitness.values = 0.0, #for maximising
                    ind.testfitness = 0.0, #for maximising     
                    countdiverr += 1
                    
    #+++++++++++++++++++++++++++++++++++++++++++++
    try:
        halloffame.update(population)
    except AttributeError:
        pass
    #+++++++++++++++++++++++++++++++++++++++++++++
    record = stats.compile(population) if stats else {}
    logbook.record(run=run, gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)  
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Capture best individual for Generation 0
    gen=0
    hof_db=[]
    cxOver_db=[] # -------------------77777777777777777777777777777777777777777
    B4Fitness = 0
    B4Test_Fitness = 0
    AfFitness = 0
    AfTest_Fitness = 0
    # -------------------------------------------------------------------------
    # Collect HOF Data  (NMSE or AdjR2) 
    if poplnType == 'Features' and FtEvlType == 'AdjR2':
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), halloffame[0].evlntime, len(halloffame[0]),
                       int(str(halloffame[0].nooffeatures)[1:-2]), float(str(halloffame[0].mser2_train)[1:-2]), float(str(halloffame[0].mser2_test)[1:-2]), str(halloffame[0])])

    elif poplnType == 'Features' and FtEvlType == 'MSE':
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness),
                       halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), str(halloffame[0])])
    # ------------------------------------------------------------------------- 
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++ Select for Replacement Function +++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    # Breeding Function - TWO OFFSPRINGS
    #+++++++++++++++++++++++++++++++++++++++++++++
    # define a breed function.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, cxOver_db, mettarget, countdiverr

        # initialise ----------------777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777
#        B4Fitness = 0
#        B4Test_Fitness = 0
#        AfFitness = 0
#        AfTest_Fitness = 0
        offspring=[]
        successCX = False
        #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        p1, p2 = list(map(toolbox.clone, tctools.selDoubleTournTime(population, 2, fitness_size=3, parsimony_size=1.4, fitness_first=False)))
        # selDoubleTournTime(individuals, k, fitness_size, parsimony_size, fitness_first, fit_attr="fitness")
#        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))
        # Fitness Before CrossOver -------------7777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777
#        B4Fitness1 = p1.fitness.values[0]
#        B4Test_Fitness1 = p1.testfitness[0]
#        
#        B4Fitness2 = p2.fitness.values[0]
#        B4Test_Fitness2 = p2.testfitness[0]
        #++++++++ Crossover +++++++++++++++++++++++++++++++++++++
        if random.random() < cxpb:
            p1, p2 = toolbox.mate(p1, p2)
            del p1.fitness.values
            del p2.fitness.values
#            successCX = True
        #   TWO OFFSPRING  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        offspring = p1,# p2
#        count=0
        for cand in offspring:  
#            str(cand)
#            count += 1
            #++++++++ mutation on the offspring ++++++++++++++++               
            if random.random() < mutpb:
                cand, = toolbox.mutate(cand)
                del cand.fitness.values
#                print(f'mutated {cand}')
            # Evaluate the offspring if it has changed
            # @@@@@@@@@@@@@@@@@(NMSE or AdjR2)@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if poplnType == 'Features' and FtEvlType == 'MSE':
                if not cand.fitness.valid:
                    #++++++++ Counting evaluations +++++++++++++++++
#                    counteval_lock.acquire()
                    counteval += 1 #Count the actual evaluations
#                    counteval_lock.release()
                    xo, yo, zo, noft = toolbox.evaluate(cand, datatrain, datatest)
                    cand.evlntime = yo,
                    cand.testfitness = zo,
                    cand.fitness.values = xo, 
                    cand.nooffeatures = noft,
                    # Check if ZeroDivisionError, ValueError 
#                    if cand.fitness.values == (0.0101010101010101,) :
#                        cand.fitness.values = 0.0, #for maximising
#                    if cand.testfitness == (0.0101010101010101,) :
#                        cand.testfitness = 0.0, #for maximising  

                    if cand.fitness.values == (0.0101010101010101,) or cand.testfitness == (0.0101010101010101,):
                        cand.fitness.values = 0.0, #for maximising
                        cand.testfitness = 0.0, #for maximising     
                        countdiverr += 1

                if float(cand.fitness.values[0]) >= target: # ---------------------------- TRAINING MSE
                    if mettarget == 0:
                        mettarget = counteval
                        print(f'Target met: {counteval}')
                        print(f'Training Fitness: {float(cand.fitness.values[0])}')
                        targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(cand.fitness.values[0]), 'Met_at': mettarget}, index = {run})
                    
                        target_csv = f'{report_csv[:-4]}_Target.csv'
                        #Export from dataframe to CSV file. Update if exists
                        if os.path.isfile(target_csv):
                            targetmet_df.to_csv(target_csv, mode='a', header=False)
                        else:
                            targetmet_df.to_csv(target_csv)   
                            
            elif poplnType == 'Features' and FtEvlType == 'AdjR2':
                if not cand.fitness.valid:
                    #++++++++ Counting evaluations +++++++++++++++++
#                    counteval_lock.acquire()
                    counteval += 1 #Count the actual evaluations
#                    counteval_lock.release()
                    xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(cand, datatrain, datatest)
#                    xo, yo, zo = toolbox.evaluate(p1)                
                    cand.evlntime = yo,
                    cand.testfitness = zo,
                    cand.fitness.values = xo, 
                    cand.nooffeatures = noft,
                    cand.mser2_train = MSER2_train, 
                    cand.mser2_test  = MSER2_test,
                    #Check if ZeroDivisionError, ValueError 
#                    if cand.fitness.values == (0.0101010101010101,) :
#                        cand.fitness.values = 0.0, #for maximising
#                    if cand.testfitness == (0.0101010101010101,) :
#                        cand.testfitness = 0.0, #for maximising  
                    if cand.fitness.values == (0.0101010101010101,) or cand.testfitness == (0.0101010101010101,):
                        cand.fitness.values = 0.0, #for maximising
                        cand.testfitness = 0.0, #for maximising     
                        countdiverr += 1                        
             #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@            

    			#[[[[[[[[[[[[[[[[[[[[[[[[[[ TARGET MET ? ]]]]]]]]]]]]]]]]]BEGIN
    			#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
   
                if float(cand.mser2_train[0]) >= target: # ---------------------------- TRAINING MSE
                    if mettarget == 0:
                        mettarget = counteval
                        print(f'Target met: {counteval}')
                        print(f'Training Fitness: {float(cand.fitness.values[0])}')
                        targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(cand.mser2_train[0]), 'Met_at': mettarget}, index = {run})
                    
                        target_csv = f'{report_csv[:-4]}_Target.csv'
                        #Export from dataframe to CSV file. Update if exists
                        if os.path.isfile(target_csv):
                            targetmet_df.to_csv(target_csv, mode='a', header=False)
                        else:
                            targetmet_df.to_csv(target_csv)  
    			#MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]END

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # REPLACEMENT - worst fitness from random k
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#            update_lock.acquire()          # LOCK !!!  
            # Identify a individual to replace from the population. Use Inverse Tournament
            candidates = selInverseTournament(population, k=1, tournsize=5)
            candidate = candidates[0]
            # Replace if offspring is better than candidate individual 
            if cand.fitness.values[0] > candidate.fitness.values[0]: # Max
            # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                    population.append(cand) 
                    population.remove(candidate)
#                    print(f'{count} replaced')
#            update_lock.release()            # RELEASE !!!
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Update hall of fame ---------------------------------------------                                                                    
            try:
                halloffame.update(population)
            except AttributeError:
                pass       
            # Record Fitness Values after Cross-Over
#            if successCX == True:
#                if count == 1:
#                    B4Fitness = B4Fitness1
#                    B4Test_Fitness = B4Test_Fitness1
#                else:
#                    B4Fitness = B4Fitness2
#                    B4Test_Fitness = B4Test_Fitness2
#                    
#                AfFitness = cand.fitness.values[0]
#                AfTest_Fitness = cand.testfitness[0]
#                
#                try:
#                    Train_imp = round(100*(AfFitness - B4Fitness)/B4Fitness)
#                except ZeroDivisionError:
#                    Train_imp = 0
#                
#                try:
#                    Test_imp  = round(100*(AfTest_Fitness - B4Test_Fitness)/B4Test_Fitness)
#                except ZeroDivisionError:
#                    Test_imp = 0
#                    
#                cxOver_db.append([run, gen, Train_imp, Test_imp])

#++++++++++++++++++++++++++++++++++++++++++++++++
#  GENERATIONAL STATs Collect Depending on Method
#++++++++++++++++++++++++++++++++++++++++++++++++
    # ----------Crossover - Features OR StdTree ----------------------------------FFFFFFFFF
    if poplnType == 'Features' and FtEvlType == 'AdjR2':
    # ----------------------------------------------------------------------------FFFFFFFFF            
        def collectStatsGen():
            nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)
            
            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), float(str(halloffame[0].mser2_train)[1:-2]), float(str(halloffame[0].mser2_test)[1:-2]), str(halloffame[0])])

    if poplnType == 'Features' and FtEvlType == 'MSE':
    # ----------------------------------------------------------------------------FFFFFFFFF            
        def collectStatsGen():
            nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)
            
            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), str(halloffame[0])])
    # -----------------------------------------------------------------------------     
    elif poplnType == 'StdTree':
    # ----------------------------------------------------------------------------FFFFFFFFF             
        def collectStatsGen():
            nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)
            
            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
            #+++++++ END OF GENERATION +++++++++++++++++++
            
#++++++++++++++++++++++++++++++++++++++++++++++++++
#  STATs for a GP Run - Collect Depending on Method
#++++++++++++++++++++++++++++++++++++++++++++++++++  
    # ----------Crossover - Features OR StdTree ----------------------------------FFFFFFFFF
    if poplnType == 'Features'  and FtEvlType == 'AdjR2':
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
            data = np.array([[*a, *b, *c, *d, *e, *f, *g] for a, b, c, d, e, f, g in zip(*data)])
            
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
            hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'No_of_Features', 'MSEr2_train', 'MSEr2_test', 'Best'])
            hof_csv = f'{report_csv[:-4]}_hof.csv'#Destination file (local)
            #Export from dataframe to CSV file. Update if exists
            if os.path.isfile(hof_csv):
                hof_dframe.to_csv(hof_csv, mode='a', header=False)
            else:
                hof_dframe.to_csv(hof_csv)
    
    # ----------Crossover - Features OR StdTree ----------------------------------FFFFFFFFF
    if poplnType == 'Features'  and FtEvlType == 'MSE':
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
            data = np.array([[*a, *b, *c, *d, *e] for a, b, c, d, e in zip(*data)])
            
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
            hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'No_of_Features','Best'])
            hof_csv = f'{report_csv[:-4]}_hof.csv'#Destination file (local)
            #Export from dataframe to CSV file. Update if exists
            if os.path.isfile(hof_csv):
                hof_dframe.to_csv(hof_csv, mode='a', header=False)
            else:
                hof_dframe.to_csv(hof_csv)      

    elif poplnType == 'StdTree':
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
            data = np.array([[*a, *b, *c, *d] for a, b, c, d in zip(*data)])
            
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
            #+++++++++++++++++++++++++++++++++++++++++++++
            #List to dataframe
            hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
            hof_csv = f'{report_csv[:-4]}_hof.csv'#Destination file (local)
            #Export from dataframe to CSV file. Update if exists
            if os.path.isfile(hof_csv):
                hof_dframe.to_csv(hof_csv, mode='a', header=False)
            else:
                hof_dframe.to_csv(hof_csv)

    #++++++++++++++++++++++++++++++++++++++++++++++++++
    #  STATs Crossover Effect
    #++++++++++++++++++++++++++++++++++++++++++++++++++  
    def collectcxOverStats():
        nonlocal run, gen, B4Fitness, B4Test_Fitness, AfFitness, AfTest_Fitness, report_csv, cxOver_db     
    #+++++++++++++++++++++++++++++++++++++++++++++
    ## Save Crossover Stats
    #++++++++++++++++++++++++++++++++++++++++++++++
        #List to dataframe
        cxOver_dframe=pd.DataFrame(cxOver_db, columns=['Run', 'Generation', 'Train_Fitness_imp', 'Test_Fitness_imp'])
#        cxOver_dframe=pd.DataFrame(cxOver_db, columns=['Run', 'Generation', 'B4Fitness', 'B4Test_Fitness', 'AfFitness', 'AfTest_Fitness'])
        cxOver_csv = f'{report_csv[:-4]}_cxOver.csv'#Destination file (local)
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(cxOver_csv):
            cxOver_dframe.to_csv(cxOver_csv, mode='a', header=False)
        else:
            cxOver_dframe.to_csv(cxOver_csv)
        cxOver_db=[]
		
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Create a Generation
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Begin the generational process
    for gen in range(1, ngen+1):
        # Generate offsprings -  equivalent to a generation / populations size
        poplnsize =  len(population)
#        poplnsize =  500
        # print(f'Number of div errors in generation {gen} = {countdiverr}')
        counteval = 0 
        countdiverr = 0
#        for h in range(poplnsize):
#            breed()
        while counteval < poplnsize:
            breed()
#            for j in range(poplnsize - counteval):
#                breed()
#        collectcxOverStats() #---------------------------------------------------FFFFFFFFF
        collectStatsGen()
    collectStatsRun()
    ###############################################################################       
    return population, logbook    
    ###############################################################################


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
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    mettarget = 0 # 0 = not set
	#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
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
    for ind in population:
        if not ind.fitness.valid:
            xo, yo, zo = toolbox.evaluate(ind, datatrain, datatest)
            ind.evlntime = yo,
            ind.testfitness = zo,
            ind.fitness.values = xo,
            if ind.fitness.values == (0.0101010101010101,) :
                ind.fitness.values = 0.0, #for maximising
            if ind.testfitness == (0.0101010101010101,) :
                ind.testfitness = 0.0, #for maximising                
#                print('check this out')
#                print(str(ind))
#                print(str(ind.fitness.values))
    #+++++++++++++++++++++++++++++++++++++++++
    # Update Hall of fame
    try:
        halloffame.update(population)
    except AttributeError:
        pass
    #+++++++++++++++++++++++++++++++++++++++++++++
    record = stats.compile(population) if stats else {}
    logbook.record(run=run, gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)  
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Capture best individual for Generation 0
    gen=0
    hof_db=[]
#    hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0])])
    hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness),
                   halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
    from operator import attrgetter
    # Replacement Tournament -----------------
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
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, target, mettarget

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
            counteval_lock.acquire()
            counteval += 1 #Count the actual evaluations
            counteval_lock.release()
            xo, yo, zo = toolbox.evaluate(p1, datatrain, datatest)
#            xo, yo, zo = toolbox.evaluate(p1)
            p1.evlntime = yo,
            p1.testfitness = zo,
            p1.fitness.values = xo, 
            #Check if ZeroDivisionError, ValueError 
            if p1.fitness.values == (0.0101010101010101,) :
                p1.fitness.values = 0.0, #for maximising
            if p1.testfitness == (0.0101010101010101,) :
                p1.testfitness = 0.0, #for maximising  
#                print('check this out')
#                print(str(p1))
#                print(str(p1.fitness.values))
                
             
 			#[[[[[[[[[[[[[[[[[[[[[[[[[[ TARGET MET ? ]]]]]]]]]]]]]]]]]BEGIN
 			#WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
            if p1.fitness.values[0] >= target: # ---------------------------- TRAINING MSE
                if mettarget == 0:
                    mettarget = counteval
                    print(f'Target met: {counteval}')
                    print(f'Training Fitness: {float(p1.fitness.values[0])}')
                    targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.fitness.values[0]), 'Met_at': mettarget}, index = {run})
                 
                    target_csv = f'{report_csv[:-4]}_Target.csv'
                    #Export from dataframe to CSV file. Update if exists
                    if os.path.isfile(target_csv):
                        targetmet_df.to_csv(target_csv, mode='a', header=False)
                    else:
                        targetmet_df.to_csv(target_csv)                    
 			#MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
 			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]END
                         
                         
                            
                            
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

    ###########################################################################
    #     Functions to Collect Stats
    ###########################################################################        
    def collectStatsGen():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
        #++++++++++ Collect Stats ++++++++++++++++++++
        record = stats.compile(population) if stats else {}
        logbook.record(run= run, gen=gen, nevals=counteval, **record)
        
        if verbose:
            print(logbook.stream) 
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Update hall of fame database for each generation
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                       halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
        
    #+++++++++++++++++++++++++++++++++++++++++++++        
    #+++++++ END OF RUN/GENERATION STATS +++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++
    def collectStatsRun():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 

        #+++++++++++++++++++++++++++++++++++++++++++++
        #    Create Report for the Run 
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Put into dataframe
        chapter_keys = logbook.chapters.keys()
        sub_chaper_keys = [c[0].keys() for c in logbook.chapters.values()]
        
        data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                     in zip(sub_chaper_keys, logbook.chapters.values())]
        data = np.array([[*a, *b, *c, *d] for a, b, c, d in zip(*data)])
        
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
        hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
        #Destination file (local)
        hof_csv = f'{report_csv[:-4]}_hof.csv'
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(hof_csv):
            hof_dframe.to_csv(hof_csv, mode='a', header=False)
        else:
            hof_dframe.to_csv(hof_csv)

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
#==============================================================================
#==============================================================================
"""



def main():
    for method in methodlist:
        
        #------- SET OPTIONS FOR THE RUNS -------------------------------------
        poolsize = ''
        if method == 'APGP': #  FtEvlType - Threadlist used!
            TCMODE = 'APGP'
            poplnType = 'Features'
            # FtEvlType = 'AdjR2'
            FtEvlType = 'MSE'
            print(F' !!!!! {method} - {TCMODE} - {FtEvlType} !!!!!')
            
        if method == 'stdGP': #no bloat control
            TCMODE = 'noTC'
            poplnType = 'StdTree'
            FtEvlType = '-' #default - tag
            print(F'!!!!! {method} - GP with Std Tree - {TCMODE} no SC !!!!!')      
            
        if method == 'ftTC1': 
            TCMODE = 'TC1' # Time Control
            poplnType = 'Features'
            FtEvlType = 'MSE'
            print(F'\n {method} - Time Control Mode = !!!!! {TCMODE} !!!!!')    
            
        if method == 'ftStd': # Features with NO SC 
            TCMODE = 'noTC'
            poplnType = 'Features'
            FtEvlType = 'MSE'
            print(F'\n !!!!! {method} - {TCMODE} - {FtEvlType} - C !!!!!')
                        
        if method == 'ftSC': # Features with SC (Double Tourn)
            TCMODE = 'noTC-DT'
            poplnType = 'Features'
            FtEvlType = 'MSE'
            print(F'\n !!!!! {method} - {TCMODE} - {FtEvlType} - SC (Double Tourn) !!!!!') 
                        
        if method == 'ftAdj': # Features with SC (Double Tourn)
            TCMODE = 'noTC'
            poplnType = 'Features'
            FtEvlType = 'AdjR2'
            print(F'\n !!!!! {method} - {TCMODE} - {FtEvlType} - AdjR2 !!!!!') 

        if method == 'ftTCAR2': # Features with SC (Double Tourn)
            TCMODE = 'TC1'
            poplnType = 'Features'
            FtEvlType = 'AdjR2'
            print(F'\n !!!!! {method} - {TCMODE} - {FtEvlType} - ft TC with  AR2 !!!!!') 

        if method == 'APGPAR2': # Features with SC (Double Tourn)
            TCMODE = 'noTC'
            poplnType = 'Features'
            FtEvlType = 'AdjR2'
            print(F'\n !!!!! {method} - {TCMODE} - {FtEvlType} - APGP with AR2 !!!!!') 



    # --------------------------------------------------------------------------------=
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, trgY), float, "x") #EnCool
        
        # ------------------------------ Features OR StdTree -------------------------------FFFFFFFFF
        if poplnType == 'Features':                   #                                     FFFFFFFFF
            def feature(left, right):
                pass
            pset.addPrimitive(feature, [float,float], float)
        else:
            if poplnType == 'StdTree':
                pass                                 #                                      FFFFFFFFF
        # ----------------------------------------------------------------------------------FFFFFFFFF
        pset.addPrimitive(operator.add, [float,float], float)
        pset.addPrimitive(operator.sub, [float,float], float)
        pset.addPrimitive(operator.mul, [float,float], float)
        pset.addPrimitive(div, [float,float], float) # ???????????????????
        pset.addPrimitive(operator.neg, [float], float)
        pset.addPrimitive(math.cos, [float], float)
        pset.addPrimitive(math.sin,[float], float)
        t = random.randint(1,100)
        pset.addEphemeralConstant(f'nrand10{method}{t}', lambda: random.randint(1,100)/20, float)  #(3a)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #Weight is positive (i.e. maximising problem)  for normalised.
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)


        """
        ===============================================================================
        # ======== TREE REPRESENTATION (FEATURES or TREES)
        ===============================================================================
        """
        #==============================================================================
        ## ======== GENERATE FEATURES
        #==============================================================================
        def poplnFt(popsize):    
            #print(f'Creating a population of individuals made of features.')
            current=[]
            # Function to check validity of ind =======================================
            def checkvalid(ind):
                validity = 'Valid'
                new = ind
                nodes, edges, labels = gp.graph(new)
                if labels[0] != 'feature':
                    validity = 'NotValid'
                else :
                    for i in range(1,len(labels)): #Range excludes root
                        #Check for FUNC nodes
                        if labels[i] == 'feature':
                            # Check parent of the FUNC node
                            for r in edges:
                                if r[1] == i:
                                    # Mark node as invalid if parent is not FUNC
                                    if labels[r[0]] != 'feature':
                                        validity = 'NotValid'
                return validity
            while len(current) < popsize:
                # 1. create ind ========================================
                newP = toolbox.population(n=1)
                new = newP[0]
                #Check validity
                if checkvalid(new) == 'NotValid':
                    #print('Individual Not Valid')
                    pass
                elif  checkvalid(new) == 'Valid' : 
                    #print('Valid')
                    current.append(new)            
            return current
        #------------------------------------------------------------------------------
        #==============================================================================
        # ======== GENERATE NON FEATURE EXPRESSION
        def genNonFt(popsize=1):    
            #print(f'Creating a population of individuals made of features.')
            current=[]
            # Function to check validity of ind =======================================
            def checkvalid(ind):
                validity = 'Valid'
                new = ind
                nodes, edges, labels = gp.graph(new)
                for i in range(len(labels)): #Range excludes root
                    #Check for FUNC nodes
                    if labels[i] == 'feature':
                        validity = 'NotValid'
                        # print(labels[i])
                return validity
            while len(current) < popsize:
                newP = toolbox.population(n=1)
                new = newP[0]
                #len(new)
                if checkvalid(new) == 'NotValid':
                    # print('non-F Not Valid')
                    pass
                else :
                    # print('non-F is -- Valid')
                    current.append(new)
            return current
        #------------------------------------------------------------------------------
        #==============================================================================
        # Extract Features from an indiviual
        def extractfeatures(individual):
            ind = individual
            nodes, edges, labels = gp.graph(ind)
            featuresE = []
            indices = []# indices of FUNC (the parent of a feature).
            for i in range(len(labels)):
                if labels[i] == 'feature':
                    indices.append(i)
            # Check child of FUNC - if feature extract     
            for l in indices:
                # Identify the two children of FUNC
                legs = []
                for r in edges:
                    if r[0] == l:
                        legs.append(r[1])
                # Check both legs of FUNC to get features
                for k in legs:
                    # If child is not a feature -> ignore
                    if labels[k] == 'feature': pass
                    # If child is a feature -> extract
                    else:
                        slice = ind.searchSubtree(k)
                        new1 = gp.PrimitiveTree(ind[:][slice])
                        featuresE.append(new1)
                        # print(str(new1))
            return featuresE
        #----------------------------------------------------------------------------

        
        #============================================================================
        #============================================================================
        #============ Collect Stats for the Final Generation ========================
        #============================================================================
        #============================================================================
        if poplnType == 'Features' and FtEvlType == 'AdjR2':
            #Function to collect stats for the last generation
            def lastgenstats(population, toolbox, gen=0,  run=0, report_csv=None, datatrain=None, datatest=None):
            #    nonlocal population, toolbox, report_csv, run, gen
                lastgen_db=[]    
                for j in range(len(population)):
                    xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(population[j], datatrain, datatest)
                    population[j].fitness.values = xo,
                    population[j].evlntime = yo,
                    population[j].testfitness = zo,
                    population[j].nooffeatures = noft,
                    population[j].mser2_train = MSER2_train, 
                    population[j].mser2_test  = MSER2_test,    
                    lastgen_db.append([run, gen, float(str(population[j].fitness)[1:-2]), float(str(population[j].testfitness)[1:-2]), float(str(population[j].evlntime)[1:-2]),
                                       len(population[j]), int(str(population[j].nooffeatures)[1:-2]), float(str(population[j].mser2_train)[1:-2]), float(str(population[j].mser2_test)[1:-2]), str(population[j])])
                lastgen_dframe=pd.DataFrame(lastgen_db, columns=['Run', 'Generation', 'Train_Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'No_of_Features', 'MSEr2_train', 'MSEr2_test', 'Best'])
                #Destination file
                lastgen_csv = f'{report_csv[:-4]}_lastgen.csv'
                #Export from dataframe to CSV file. Update if exists               
                if os.path.isfile(lastgen_csv):
                    lastgen_dframe.to_csv(lastgen_csv, mode='a', header=False)
                else:
                    lastgen_dframe.to_csv(lastgen_csv)
        # =====================================================================       
        if poplnType == 'Features' and FtEvlType == 'MSE':
            #Function to collect stats for the last generation
            def lastgenstats(population, toolbox, gen=0,  run=0, report_csv=None, datatrain=None, datatest=None):
            #    nonlocal population, toolbox, report_csv, run, gen
                lastgen_db=[]    
                for j in range(len(population)):
                    xo, yo, zo, noft = toolbox.evaluate(population[j], datatrain, datatest)
                    population[j].fitness.values = xo,
                    population[j].evlntime = yo,
                    population[j].testfitness = zo,
                    population[j].nooffeatures = noft,
                    # population[j].mser2_train = MSER2_train, 
                    # population[j].mser2_test  = MSER2_test,    
                    lastgen_db.append([run, gen, float(str(population[j].fitness)[1:-2]), float(str(population[j].testfitness)[1:-2]), float(str(population[j].evlntime)[1:-2]),
                                       len(population[j]), int(str(population[j].nooffeatures)[1:-2]), str(population[j])])
                lastgen_dframe=pd.DataFrame(lastgen_db, columns=['Run', 'Generation', 'Train_Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'No_of_Features', 'Best'])
                #Destination file
                lastgen_csv = f'{report_csv[:-4]}_lastgen.csv'
                #Export from dataframe to CSV file. Update if exists               
                if os.path.isfile(lastgen_csv):
                    lastgen_dframe.to_csv(lastgen_csv, mode='a', header=False)
                else:
                    lastgen_dframe.to_csv(lastgen_csv)
        # =====================================================================
        elif poplnType == 'StdTree':
            #Function to collect stats for the last generation
            def lastgenstats(population, toolbox, gen=0,  run=0, report_csv=None, datatrain=None, datatest=None):
            #    nonlocal population, toolbox, report_csv, run, gen
                lastgen_db=[]    
                for j in range(len(population)):
                    xo, yo, zo = toolbox.evaluate(population[j], datatrain, datatest)
                    population[j].fitness.values = xo,
                    population[j].evlntime = yo,
                    population[j].testfitness = zo,
                    lastgen_db.append([run, gen, float(str(population[j].fitness)[1:-2]), float(str(population[j].testfitness)[1:-2]), float(str(population[j].evlntime)[1:-2]), len(population[j]), str(population[j])])
                lastgen_dframe=pd.DataFrame(lastgen_db, columns=['Run', 'Generation', 'Train_Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
                #Destination file
                lastgen_csv = f'{report_csv[:-4]}_lastgen.csv'
                #Export from dataframe to CSV file. Update if exists
                if os.path.isfile(lastgen_csv):
                    lastgen_dframe.to_csv(lastgen_csv, mode='a', header=False)
                else:
                    lastgen_dframe.to_csv(lastgen_csv)
        #======================================================================
        #======================================================================

        #-------------------------------------------------------------------------
        #--------- EVALUATION   v5 (predict and MSE) -------------------------------- EVALUATION   v5 (predict and MSE)
        #-------------------------------------------------------------------------
        def evalfeat(individual, datatrain, datatest):
            time_st = time.perf_counter() # <-- start timing the evaluation
        #-------- Extract features from individual ---------------
            featofInd = extractfeatures(individual)
            ftresult = pd.DataFrame()
            # If ZeroDivisionError or ValueError assign worst fitness
            try:                
                for j in range(0,len(featofInd)): 
                    func = toolbox.compile(expr=featofInd[j])
                    resultlist = []
                    # Evaluate Feature j with data
                    for item in datatrain:
                        Iresult = func(*item[:trgY])
                        resultlist.append(Iresult)
                    # Create Column and add result for feature j
                    ftresult[f'x{j}']=resultlist
                # Get True Y ------------------------------------------------------
                Y_actual = []
                for item in datatrain:
                    Y_actual.append(item[trgY])
                # Append True Y ---------------------------------------------------
                ftresult['Y'] = Y_actual
                X = ftresult.loc[:, ftresult.columns != 'Y']
                Y = ftresult['Y']
                #-------------------------------------------
                # Create linear regression object.
                mlr= LinearRegression()
                #------------------------------------------- 
                # Fit linear regression.
                mlr.fit(X, Y)
                # Predict Y - Training Data -----------------------------------
                y_pred = mlr.predict(X) 
                #-------------------------------------------
                MSE = metrics.mean_squared_error(Y_actual, y_pred)
                error = 1/(1+ MSE)          
                # Evaluate Features with TEST data=====================================
                t_ftresult = pd.DataFrame()
                for j in range(0,len(featofInd)): 
                    func = toolbox.compile(expr=featofInd[j])
                    t_resultlist = []
                    for item in datatest:
                        Iresult = func(*item[:trgY])
                        t_resultlist.append(Iresult)
                    # Create Column and add result for feature j      
                    t_ftresult[f'x{j}']=t_resultlist
                # Get Actual Y --------------------------------------------------------
                tY_actual = []
                for item in datatest:
                    tY_actual.append(item[trgY])
                #----------------------------------------------------------------------
                t_ftresult['Y'] = tY_actual
                X = t_ftresult.loc[:, t_ftresult.columns != 'Y']
                Y = t_ftresult['Y']
                #-------------------------------------------
                # Predict
                ty_pred = mlr.predict(X) 
                t_MSE = metrics.mean_squared_error(tY_actual, ty_pred)
                error_test = 1/(1+ t_MSE) 
            except (ZeroDivisionError, ValueError, TypeError):#    except ZeroDivisionError:
                error = 0.010101010101010101
                error_test = 0.010101010101010101  
        #        print('zero/value error caught')              
            evln_sec=float((time.perf_counter() - time_st))
        #    print('MSE  MSE  MSE  MSE')
        
            return error, evln_sec, error_test, len(featofInd)
        
        #-------------------------------------------------------------------------
        #---------   EVALUATION   STD Tree           -------------------------------- EVALUATION   STD Tree 
        #-------------------------------------------------------------------------    
        #    Evaluate the mean squared error between the expression
        def evalSymbReg(individual, datatrain, datatest):
            # Transform the tree expression in a callable function
            func = toolbox.compile(expr=individual)
            time_st = time.perf_counter() # <-- start reading after compilation
            # Evaluate the mean squared error between the expression and the real function
            #Training Error - Fitness ===============================
            for z in range(2): #                                                       
                error=0.
                total=0.
                try:
                    for item in datatrain:
                        total = total + ((func(*item[:trgY])) - (item[trgY]))**2
                        MSE = total/len(datatrain)
                        error = 1/(1+ MSE)       # Normalise        
                except (ZeroDivisionError, ValueError, TypeError):#    except ZeroDivisionError:
                        error = 0.010101010101010101
            #Test Data =============================================
            error_test=0.
            total_t=0.
            try:
                for item in datatest:
                    total_t = total_t + ((func(*item[:trgY])) - (item[trgY]))**2    
                    MSE_t = total_t/len(datatest)
                    error_test = 1/(1+ MSE_t)               
         
            except (ZeroDivisionError, ValueError, TypeError):#    except ZeroDivisionError:
                    error_test = 0.010101010101010101
                    
            evln_sec=float((time.perf_counter() - time_st))
            return error, evln_sec, error_test
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        #--------- EVALUATION   v4 (adjusted R^2) ----------------------------------- EVALUATION   v4 (adjusted R^2) 
        #-------------------------------------------------------------------------
        def evalfeatAjR2(individual, datatrain, datatest):
            time_st = time.perf_counter() # <-- start timing the evaluation
        #-------- Extract features from individual ---------------
            featofInd = extractfeatures(individual)
            ftresult = pd.DataFrame()
            # If ZeroDivisionError or ValueError assign worst fitness
            MSE_R2_error = 0
            MSE_R2_error_test = 0       
            try:     
             
                for j in range(0,len(featofInd)): 
                    func = toolbox.compile(expr=featofInd[j])
                    resultlist = []
                    # Evaluate Feature j with data
                    for item in datatrain:
                        Iresult = func(*item[:trgY])
                        resultlist.append(Iresult)
                    # Create Column and add result for feature j
                    ftresult[f'x{j}']=resultlist
                # Get True Y ----------------------------------------------------------
                Y_actual = []
                for item in datatrain:
                    Y_actual.append(item[trgY])
                # Append True Y -------------------------------------------------------
                ftresult['Y'] = Y_actual
                X = ftresult.loc[:, ftresult.columns != 'Y']
                Y = ftresult['Y']
                #-------------------------------------------
                # Create linear regression object.
                mlr= LinearRegression()
                # Fit linear regression -------------------- 
                mlr.fit(X, Y)
                # Get R^2 square and adjusted R^2.
                rsqTrain = mlr.score(X, Y)
                adj_rsqTrain = 1-(1-rsqTrain)*((len(datatrain)-1)/(len(datatrain)-len(featofInd)-1))
                # If adjusted r^2 is negative assign worst fitness 0
                if adj_rsqTrain < 0:
                    error = 0
                    # print(f'adj_rsqTrain {adj_rsqTrain}')
                else :
                    error = adj_rsqTrain
                    
                # Get NMSE --------------------- Training ------------#################
                ym_pred = mlr.predict(X) 
                MSE_R2 = metrics.mean_squared_error(Y_actual, ym_pred)
                MSE_R2_error = 1/(1+ MSE_R2)          
                #-------------------------------------------        
                
                # Evaluate Features with TEST data-----------------------------########
                t_ftresult = pd.DataFrame()
                for j in range(0,len(featofInd)): 
                    func = toolbox.compile(expr=featofInd[j])
                    t_resultlist = []
                    for item in datatest:
                        Iresult = func(*item[:trgY])
                        t_resultlist.append(Iresult)
                    # Create Column and add result for feature j      
                    t_ftresult[f'x{j}']=t_resultlist
                # Get Actual Y ------------------------------------------------########
                tY_actual = []
                for item in datatest:
                    tY_actual.append(item[trgY])
                #--------------------------------------------------------------########
                t_ftresult['Y'] = tY_actual
                X = t_ftresult.loc[:, t_ftresult.columns != 'Y']
                Y = t_ftresult['Y']
                #-Predict------------------------------------------------------########
                rsqTest = mlr.score(X, Y)
                adj_rsqTest = 1-(1-rsqTest)*((len(datatrain)-1)/(len(datatrain)-len(featofInd)-1))
                # If adjusted r^2 is negative assign worst fitness 0
                if adj_rsqTest < 0:
                    error_test = 0
                else :
                    error_test = adj_rsqTest
                #--------------------------------------------------------------########
                # Get NMSE --------------------- Testing ----------------------########
                tym_pred = mlr.predict(X) 
                t_MSE_R2 = metrics.mean_squared_error(tY_actual, tym_pred)
                MSE_R2_error_test = 1/(1+ t_MSE_R2) 
            #------------------------------------------------------------------########
            except (ZeroDivisionError, ValueError, TypeError):#    except ZeroDivisionError:
                error = 0.010101010101010101
                error_test = 0.010101010101010101                
            evln_sec=float((time.perf_counter() - time_st))
            # print('adjusted R2      adjusted R2')

            return error, evln_sec, error_test, len(featofInd), MSE_R2_error, MSE_R2_error_test
        #----------------------------------------------------------------------########

        #======================================================================
        #======================================================================
        #             CLASSIFICATION -----!
        # from sklearn.preprocessing import StandardScaler
        from sklearn import preprocessing

        # min_max_scaler = preprocessing.MinMaxScaler()
        # X_train_minmax = min_max_scaler.fit_transform(X_train)
        # X_train_minmax
        #======================================================================
        #======================================================================
        if PREDSYS == 'CLASSIFICATION':
            # individual = pop[2]
            #-------------------------------------------------------------------------
            #--------- Feature - CLASSIFICATION Evaluation   
            #-------------------------------------------------------------------------
            # def evalFeatClsfn(individual, datatrain, datatest):
            def evalfeat(individual, datatrain, datatest):
                time_st = time.perf_counter() # <-- start timing the evaluation
                
                
            #-------- Extract features from individual ---------------
                featofInd = extractfeatures(individual)
                
                ftresult = pd.DataFrame()
                # If ZeroDivisionError or ValueError assign worst fitness
                try:                
                    for j in range(0,len(featofInd)): 
                        func = toolbox.compile(expr=featofInd[j])
                        resultlist = []
                        # if div error assign zeros to result of fetature
                        try:
                            # Evaluate Feature j with data
                            for item in datatrain:
                                Iresult = func(*item[:trgY])
                                resultlist.append(Iresult)
                        except (ZeroDivisionError, ValueError, TypeError):
                            resultlist = []
                            for item in datatrain:
                                Iresult = 1
                                resultlist.append(Iresult)                        
                            # --------------------------------------------
    
                        # Create Column and add result for feature j
                        ftresult[f'x{j}']=resultlist
                        
                        
                        
                        
                    # normalise / scale the data  {{{{{{{{{------------------------ 
                    min_max_scaler = preprocessing.MinMaxScaler()
                    test = min_max_scaler.fit_transform(ftresult)
                    test2 = pd.DataFrame(test)
                    test2.columns=ftresult.columns
                    ftresult=test2
                    #-----------------------------------------------------}}}}}}}}}      
                        
                    
                    
                    # Get True Y --------------------------------------------------------------------
                    Y_actual = []  # len(Y_actual)
                    for item in datatrain:
                        Y_actual.append(item[trgY])
    #               len(Y_actual)                    
    
                    #--------------------------------------------------------------------------------               
                    #-------------LogisticRegression-------------------------------------------------
                    Clsfy = LogisticRegression(random_state=0)
    
                    # # +++++++DecisionTreeClassifier++++++++++++++++++++++++++
                    # Clsfy = DecisionTreeClassifier(random_state=0)
                    
                    # # +++++++GaussianNaiveBayes++++++++++++++++++++++++++++++   
                    # Clsfy = GaussianNB()
                    
                    # # ++++++KNeighborsClassifier+++++++++++++++++++++++++++++   
                    # Clsfy=KNeighborsClassifier()
                    
                    # # ++++++SupportVectorClassifier++++++++++++++++++++++++++  
                    # Clsfy=SVC()
                    
                    # # ++++++RandomForestClassifier+++++++++++++++++++++++++++
                    # Clsfy=RandomForestClassifier()
                                    
                    Clsfy.fit(ftresult,Y_actual)
                    DTy_pred = Clsfy.predict(ftresult)
                    MSE = metrics.mean_squared_error(Y_actual, DTy_pred)
                    error = 1/(1+ MSE)  
                    # DT_clf.score(ftresult,Y_actual)
    
                    # TEST data=====================================
                    t_ftresult = pd.DataFrame()
                    for j in range(0,len(featofInd)): 
                        # func = toolbox.compile(expr=featofInd[j])
                        t_resultlist = []
                        
                        # for item in datatest:
                        #     Iresult = func(*item[:trgY])
                        #     t_resultlist.append(Iresult)
                            
                            
                        try:
                            # Evaluate Feature j with data
                            for item in datatest:
                                Iresult = func(*item[:trgY])
                                t_resultlist.append(Iresult)
                                # print(Iresult)
                                # print(item)
                                
                        except (ZeroDivisionError, ValueError, TypeError):
                            resultlist = []
                            for item in datatrain:
                                Iresult = 1
                                t_resultlist.append(Iresult)            
                                                         
                            # --------------------------------------------                    
                        # Create Column and add result for feature j      
                        t_ftresult[f'x{j}']=t_resultlist

                        
                    # normalise / scale the data  {{{{{{{{{------------------------ 
                    min_max_scaler = preprocessing.MinMaxScaler()
                    test = min_max_scaler.fit_transform(t_ftresult)
                    test2 = pd.DataFrame(test)
                    test2.columns=t_ftresult.columns
                    t_ftresult=test2
                    #-----------------------------------------------------}}}}}}}}}      

                    
                    # Get Actual Y --------------------------------------------------------
                    tY_actual = []
                    for item in datatest:
                        tY_actual.append(item[trgY])
                        
                    # # Predict -----------------------------------
                    ty_pred = Clsfy.predict(t_ftresult)                
                                    
                    # ty_pred = mlr.predict(X) 
                    t_MSE = metrics.mean_squared_error(tY_actual, ty_pred)
                    error_test = 1/(1+ t_MSE) 
                    
                    ty_pred=Clsfy.predict(t_ftresult)
                    t_MSE = metrics.mean_squared_error(tY_actual, ty_pred)
                    
                    # LRclf.score(t_ftresult,tY_actual)#----------------------------------------??????????????????????
                    # DT_clf.score(t_ftresult,tY_actual)#----------------------------------------??????????????????????
        
                except (ZeroDivisionError, ValueError, TypeError):#    except ZeroDivisionError:
                    error = 0.010101010101010101
                    error_test = 0.010101010101010101  
                    # print('zero/value error caught')    
                    
                    
                evln_sec=float((time.perf_counter() - time_st))
                # print('MSE  MSE  MSE  MSE')
            
                return error, evln_sec, error_test, len(featofInd)
            
# --------------------------------------------


        ######################################
        # GP Mutations                       #
        ######################################
        #def mutUniformFT(individual, expr, pset):
        def mutUniformFT(individual):#, pset):
            """Randomly select a point in the tree *individual*, then replace the
            subtree at that point as a root by the expression generated using method
            :func:  'poplnFt(1)' OR 'genNonFt(1)'.
            :param individual: The tree to be mutated.
            :param expr: A function object that can generate an expression when called.
            :returns: A tuple of one tree.
            """
        # individual = v_ind[8]
            sub=[]
            # Choose to mutate with a feature or sub-featue expression
            if random.random() < 0.5:
                #Create a feature
                sub = poplnFt(1)[0]
                subType ='feature'
            else:
                #create a sub-feature
                sub = genNonFt(1)[0]
                subType ='notFUNC'
            # print(f'Subtree Type: {subType}')
            nodes, edges, labels = gp.graph(individual) 
            #-------------------------------------------------------------------- 
            count = 0
            success = False
            while count < 10 and success == False:
                count += 1
                index = random.randrange(1,len(individual)) # Leave the root node 
                #print(index)
                labels[index]
                edges[index-1][0]
                pointparent = ''
                # Check selected node type 'feature' or 'non-feature'
                if labels[index] == 'feature':
                    pointtype = 'feature' 
                else:
                    pointtype = 'non-feature'
                #Check point parent type if point not FUNC 
                if pointtype != 'feature':
                    #check parenttype
                    #print(edges[index-1])
                    parent = edges[index-1][0]
                    pointparent = labels[parent]
                # print(f'Point Type: {subType}')
                # Substitute if transaction is valid i.e. Type for type OR different and valid
                if (pointtype == subType) or (pointtype != subType and pointparent == 'feature'):
                    slice_ = individual.searchSubtree(index)
                    individual[slice_] = sub
                    success = True
                # checkvalid(individual) # ---------------???????????
            return individual,
        


        """ 
        =========================================================================
        ====================    INITIALISATION 3  ===============================
        =========================================================================
        """
        # -----------------------------------------------------------------------FFFFFFFFF  
        # -----------Evaluation - Features OR StdTree ---------------------------FFFFFFFFF
        # -----------------------------------------------------------------------FFFFFFFFF  
        if poplnType == 'Features' and FtEvlType == 'MSE':
            toolbox.register("evaluate", evalfeat)
        elif poplnType == 'StdTree':
            toolbox.register("evaluate", evalSymbReg)
        elif poplnType == 'Features' and FtEvlType == 'AdjR2':
            toolbox.register("evaluate", evalfeatAjR2)
            
        # -----------------------------------------------------------------------FFFFFFFFF  
        # ----------Crossover - Features OR StdTree -----------------------------FFFFFFFFF
        # -----------------------------------------------------------------------FFFFFFFFF  
        if poplnType == 'Features':
            toolbox.register("mate", cxOnePointFt)
        elif poplnType == 'StdTree':
            toolbox.register("mate", gp.cxOnePoint)

        # -----------------------------------------------------------------------FFFFFFFFF  
        # ------------------- Mutation - Features OR StdTree --------------------FFFFFFFFF
        # -----------------------------------------------------------------------FFFFFFFFF  
        if poplnType == 'Features':
            toolbox.register("mutate", mutUniformFT)
        elif poplnType == 'StdTree':
            toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
            toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        # -----------------------------------------------------------------------FFFFFFFFF  
        #toolbox.register("select", tools.selTournament, tournsize=3) # -------------------?? breeding
        #toolbox.register("select", tools.selDoubleTournament) # --------------------------?? breeding
        #toolbox.register("select", tctools.selDoubleTournTFL2) # -------------------------?? breeding

        # -----------------------------------------------------------------------FFFFFFFFF  
        # -------------------  Selection --------------------------------------------
        # -----------------------------------------------------------------------FFFFFFFFF  
        if TCMODE == 'TC1' or TCMODE == 'TC2' or TCMODE == 'TC3': # Replace Size with Time only
            toolbox.register("select", tctools.selDoubleTournTime)
        
        elif TCMODE == 'noTC'  : #==========or TCMODE == 'APGP' or TCMODE == 'APGP50'====????????????????????????????????????????????????!!!!!22222222222222222222
        # elif TCMODE == 'noTC' or TCMODE != 'APGP' : #===or TCMODE != 'APGP50'==========????????????????????????????????????????????????!!!!!22222222222222222222
            toolbox.register("select", tools.selTournament, tournsize=3) # ----------------------------?? breeding
            # toolbox.register("select", tools.selDoubleTournament)       
            
        elif TCMODE == 'noTC-DT'  : #==========or TCMODE == 'APGP' or TCMODE == 'APGP50'====????????????????????????????????????????????????!!!!!22222222222222222222
        # elif TCMODE == 'noTC' or TCMODE != 'APGP' : #===or TCMODE != 'APGP50'==========????????????????????????????????????????????????!!!!!22222222222222222222
            toolbox.register("select", tools.selDoubleTournament) 
            # p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2, fitness_size=3, parsimony_size=1.4, fitness_first=False)))

        elif TCMODE == 'APGP' or TCMODE == 'APGP50':
            # toolbox.register("select", tools.selDoubleTournament)       
            toolbox.register("select", tools.selTournament, tournsize=3) # ----------------------------?? breeding


        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        # When an over the limit child is generated, it is simply replaced by a randomly selected parent.
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.register("worstfitness", tools.selWorst)
        # -----------------------------------------------------------------------------


        """
        =================================================================================
        Function to create initial population: (1) FIXED SIZE AND (2)  UNIQUE INDIVIDUALS
        (Constants are treated as same).
        =================================================================================
        """
        def inipoplnF(popsize):    
            # nonlocal poplnType #---------------------------------------------------------FFFFFFFFF
            ini_len = FLISize # Initial lengths
            # popsize = 500
            print(f'===========================================================================')
            print(f'\n Creating a population of {popsize} individuals (FEATURES) - each of size: {ini_len}')
            # Function to extract the node types   ---------------------------------------FFFFFFFFF
            def graph(expr):
                str(expr)
                nodes = range(len(expr))
                edges = list()
                labels = dict()
                stack = []
                for i, node in enumerate(expr):
                    if stack:
                        edges.append((stack[-1][0], i))
                        stack[-1][1] -= 1
                    labels[i] = node.name if isinstance(node, gp.Primitive) else node.value
                    stack.append([i, node.arity])
                    while stack and stack[-1][1] == 0:
                        stack.pop()
                return nodes, edges, labels
            # --------------------------------------------------------------- create 1st individual
            current=[]
            newind=[]
            # ----------------------------------------------------------------------------FFFFFFFFF
            # newind= toolbox.population(n=1)
            if poplnType == 'Features':
                newind = poplnFt(1)# 
            elif poplnType == 'StdTree':
                newind = toolbox.population(1)# 
                # ------------------------------------------------------------------------FFFFFFFFF            
            while len(newind[0]) != ini_len:
                newind = toolbox.population(n=1)
            current.append(newind[0])
            # ------------------------------- Create others; 
            # For each new one check to see a similar individual exists in the population.
            while len(current) < popsize:
            # ----------------------------------------------------------------------------FFFFFFFFF
            #    newind= toolbox.population(n=1)
                if poplnType == 'Features':
                    pop = poplnFt(1)# 
                elif poplnType == 'StdTree':
                    pop = toolbox.population(1)# 
                    # --------------------------------------------------------------------FFFFFFFFF         
                if len(pop[0]) == ini_len:
                    # ----------------------------- Check for duplicate
                    lnodes, ledges, llabels = graph(pop[0])
                    similarity = 'same'
                    for k in range(len(current)): # CHECK all INDs in CURRENT population
                        nodes, edges, labels = graph(current[k])
                        for j in range(len(labels)): # Check NEW against IND from CURRENT
                            constants = 'no' # will use to flag constants
                            if labels[j] != llabels[j]: 
                                similarity = 'different' 
                                # no need to check other nodes as soon as difference is detected 
                            if '.' in str(labels[j]) and '.' in str(llabels[j]): constants = 'yes'
                            if labels[j] != llabels[j] or constants != 'yes': # They are different and not constants
                                continue # no need to check other nodes as soon as difference is detected 
                        if similarity =='same': # skips other checks as soon as it finds a match
                            continue
                    if similarity == 'different': # add only if different from all existing
                        current.append(pop[0])     
            print('population created')
            return current
        """
        ============================================================================
        ============================================================================
        """
        
        """
        ============================================================================
        Function to create initial population: (1) FIXED SIZE AND (2)  UNIQUE INDIVIDUALS
        (Constants are treated as same).
        ============================================================================
        """
        def inipopln(popsize):    
            ini_len = 10 # Initial lengths
        #    popsize = 500
            print(f'Creating a population of {popsize} individuals - each of size: {ini_len}')
            # Function to extract the node types   ---------------------------------------FFFFFFFFF
            def graph(expr):
                str(expr)
                nodes = range(len(expr))
                edges = list()
                labels = dict()
                stack = []
                for i, node in enumerate(expr):
                    if stack:
                        edges.append((stack[-1][0], i))
                        stack[-1][1] -= 1
                    labels[i] = node.name if isinstance(node, gp.Primitive) else node.value
                    stack.append([i, node.arity])
                    while stack and stack[-1][1] == 0:
                        stack.pop()
                return nodes, edges, labels
        #    ------------------------------- create 1st individual
            current=[]
            newind=[]
            newind= toolbox.population(n=1)
            while len(newind[0]) != ini_len:
                newind = toolbox.population(n=1)
            current.append(newind[0])
        #    ------------------------------- Create others; 
        #    For each new one check to see a similar individual exists in the population.
            while len(current) < popsize:
                pop = toolbox.population(n=1)
                if len(pop[0]) == ini_len:
                    # ----------------------------- Check for duplicate
                    lnodes, ledges, llabels = graph(pop[0])
                    similarity = 'same'
                    for k in range(len(current)): # CHECK all INDs in CURRENT population
                        nodes, edges, labels = graph(current[k])
                        for j in range(len(labels)): # Check NEW against IND from CURRENT
                            constants = 'no' # will use to flag constants
                            if labels[j] != llabels[j]: 
                                similarity = 'different' 
                                # no need to check other nodes as soon as difference is detected 
                            if '.' in str(labels[j]) and '.' in str(llabels[j]): constants = 'yes'
                            if labels[j] != llabels[j] or constants != 'yes': # They are different and not constants
                                continue # no need to check other nodes as soon as difference is detected 
                        if similarity =='same': # skips other checks as soon as it finds a match
                            continue
                    if similarity == 'different': # add only if different from all existing
                        current.append(pop[0])     
            print('population created')
            return current
        """
        ===============================================================================
        ===============================================================================
        """
    
    
        random.seed(2020)
        
    # =============================================================================
        tag = f'{TCMODE}_{poplnType}_{FtEvlType}_{dataPLMB}'[:-4]#Standard GP - Steady State FtEvlType poplnType
    # -----------------------------------------------------------------------
        if system == 'server':
            reportfolder = "/home/aliyu/Documents/Features/PLMBreport/set2\\"
        elif system == 'laptop':
            reportfolder = f"C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Ph5\\02_FEATURES\\"
        elif system == 'desktop':
            reportfolder = f"C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Ph5\\02_FEATURES\\PLMBreport\\set2\\"
        elif system == 'NBOOK':  
            reportfolder = f"C:\\Users\\Aliyu Sambo\\OneDrive - Birmingham City University\\Experiment_Ph6\\02_FEATURES\\classificationGPML\\"
            # reportfolder = f"C:\\Users\\Aliyu Sambo\\OneDrive - Birmingham City University\\Experiment_Ph5\\02_FEATURES\\"
        elif system == 'bsystem':
            reportfolder = f"C:\\Users\\PC\\Documents\\Features\\PLMBreport\\"
        elif system == 'Dktop2':
            reportfolder = f"C:\\Users\\user\\Documents\\02_FEATURES\\\classificationGPML\\GPMLResult2\\"

        if method != 'APGP' or method != 'APGPAR2':
            report_csv = f"{reportfolder}PH6_{run_time}_{tag}.csv"
      # ---------------------------------------------------------------------------
        for i in range(1, runs+1):
            run = i
   
            if method != 'APGP' and method != 'APGPAR2': # i.e. wait till later to create initial population
                if poplnType == 'Features':
                    pop = inipoplnF(popsize)# FLI with Features
                    
                elif poplnType == 'StdTree':
                    pop = inipopln(popsize)# FLI with std tree                    

                    
            # ------------------------------------------------------------------------FFFFFFFFF 
            # ======================  Sorting the Population   ====================
            # pop = inipopln(500)
            # pop = list(sorted(pop, key=lambda i: len(i)))#-------------!!!!!!!!!! FL ------------------------FL----------------FL
            # print('sorted: ------------- ')
            # for i in range(0,len(pop)): print(f'{i} - length: {len(pop[i])} - {pop[i]}')
            # ---------------------------------------------------------------------
            hof = tools.HallOfFame(1) 
            # -----------------------------------------------------------------------FFFFFFFFF  
            #               Stats
            # -----------------------------------------------------------------------FFFFFFFFF  
            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
            stats_size = tools.Statistics(len)
            stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
            stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
            # Configure Stats accordingly -----------------------------------------
            if poplnType == 'Features' and FtEvlType == 'AdjR2':
                stats_nooffeatures = tools.Statistics(lambda ind: ind.nooffeatures)
                stats_mser2_train = tools.Statistics(lambda ind: ind.mser2_train)
                stats_mser2_test  = tools.Statistics(lambda ind: ind.mser2_test)
                mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness, nooffeatures=stats_nooffeatures,
                                               mser2_train=stats_mser2_train, mser2_test=stats_mser2_test)
                print('\n ====>>')
#                print('Working with FEATURES and Adjusted R2 for fitness...')
            elif poplnType == 'Features' and FtEvlType == 'MSE':   
                stats_nooffeatures = tools.Statistics(lambda ind: ind.nooffeatures)
                mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness, nooffeatures=stats_nooffeatures)
                print('\n ====>>')
                print('Working with FEATURES and Normalised Mean Squared Error (NMSE) for fitness...')
            elif poplnType == 'StdTree':  
                mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness)            
                print('\n ====>>')
                print('Working with STANDARD Tree and Normalised Mean Squared Error (NMSE) for fitness...')
            # ---------------------------------------------------------------------
            mstats.register("avg", numpy.mean)
            mstats.register("std", numpy.std)
            mstats.register("min", numpy.min)
            mstats.register("max", numpy.max)




            if poplnType == 'StdTree':
                print(f'{TCMODE} on Features -------------!')  
                pop, log = gpSteadyState(pop, toolbox, 0.9, 0.1, nofGen, #              (9)                             
                # pop, log = gpSteadyState(pop, toolbox, 0.9, 0.1, nofGen, #              (9)
                                         stats=mstats, halloffame=hof, verbose=False, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target = settarget)#,poplnType =poplnType,FtEvlType =FtEvlType)#0.025
                print(f'Not Time-Control on Features -------------!')   


            elif poplnType == 'Features'  and TCMODE == 'TC1' and method != 'APGPAR2': # i.e. for TC1(plain time control) and TC2(time control with FLI)[popln initialisation differentiates, see above.]
                if TCMODE == 'TC1': print(f' Time Control: {TCMODE} on Features')
                pop, log = gpDoubleTC(pop, toolbox, 0.9, 0.1, nofGen, # 
                                         stats=mstats, halloffame=hof, verbose=False, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target = settarget,poplnType =poplnType,FtEvlType =FtEvlType)#0.025
      

            elif poplnType == 'Features'  and method == 'ftStd': # 
                print(f'{TCMODE} on Features -------------!')               
                pop, log = gpFTStd(pop, toolbox, 0.9, 0.1, nofGen, # 
                                         stats=mstats, halloffame=hof, verbose=False, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target = settarget,poplnType =poplnType,FtEvlType =FtEvlType)#0.025

            elif poplnType == 'Features'  and method == 'ftSC': # 
                print(f'{TCMODE} on Features -------------!')               
                pop, log = gpDoubleCx(pop, toolbox, 0.9, 0.1, nofGen, # 
                                         stats=mstats, halloffame=hof, verbose=False, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target = settarget,poplnType =poplnType,FtEvlType =FtEvlType)#0.025
                     
            elif poplnType == 'Features'  and method == 'ftAdj': # poplnType = 'Features'  ; method = 'ftAdj' ; method = 'APGPAR2'
                print(f'{TCMODE} on Features -------------!')               
                pop, log = gpFTStd(pop, toolbox, 0.9, 0.1, nofGen, # 
                                         stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target = settarget,poplnType =poplnType,FtEvlType =FtEvlType)#0.025

            # if poplnType == 'Features' and TCMODE == 'TC3': # i.e. (time control with FLI + neighbours)
            #     print(f' Time Control: {TCMODE} - FLI, Sorted, Neighbours Breeding')
            #     pop, log = gpDoubleCxFTC3(pop, toolbox, 0.9, 0.1, nofGen, # 
            #                              stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target = settarget,poplnType =poplnType,FtEvlType =FtEvlType)#0.025

            # elif poplnType == 'Features'  and TCMODE != 'TC3' and TCMODE != 'APGP' and TCMODE != 'APGP50': # i.e. for TC1(plain time control) and TC2(time control with FLI)[popln initialisation differentiates, see above.]
            #     if TCMODE == 'TC2': print(f' Time Control: {TCMODE} - FLI')
            #     if TCMODE == 'TC1': print(f' Time Control: {TCMODE} on Features')
            #     pop, log = gpDoubleCx(pop, toolbox, 0.9, 0.1, nofGen, # 
            #                              stats=mstats, halloffame=hof, verbose=False, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target = settarget,poplnType =poplnType,FtEvlType =FtEvlType)#0.025
            #                             # stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest, target = settarget,poplnType =poplnType,FtEvlType =FtEvlType)#0.025

            # elif poplnType == 'Features'  and TCMODE == 'APGP50': # 
            #     poolsize = 50
            #     print(f'{TCMODE} on Features -------------!')               
            #     pop, log = APGPFtL(pop, toolbox, 0.9, 0.1, nofGen, # 
            #                              stats=mstats, halloffame=hof, verbose=False, run=run, report_csv=report_csv, poolsize=poolsize, datatrain=datatrain, datatest=datatest, target = settarget,poplnType =poplnType,FtEvlType =FtEvlType)#0.025

            # -----------------------------------------------------------------FFFFFFFFF  
            # for i in range(0,len(pop)): print(f'{i} - length: {len(pop[i])} - {pop[i]}')
            if method != 'APGP' and method != 'APGPAR2':

                print(f'Taking stats for the last generation....Run: {run}')
                #Collect stats for the last generation of each run.
                lastgenstats(pop, toolbox, gen=nofGen, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest)#GEN....??  (9b)
            # =================================================================
                print(f'.... {dataPLMB} ....-Run = {run}')
            
# -----------------------------------------------------------------------------
#     ---------------   APGP  ------------------
# -----------------------------------------------------------------------------
        if poplnType == 'Features'  and TCMODE == 'APGP' and method != 'APGPAR2': # 
            for poolsize in THREADLIST:
                
                for i in range(1, runs+1):
                    run = i
                    pop = inipoplnF(popsize)
                    print(f'APGP-{poolsize} on Features --------------Run = {run}!')               
                    tag = f'APGP{poolsize}_{poplnType}_{FtEvlType}_{dataPLMB}'[:-4]#Standard GP - Steady State FtEvlType poplnType
                    report_csv = f"{reportfolder}PH6_{run_time}_{tag}.csv"

                    pop, log = APGPFtL(pop, toolbox, 0.9, 0.1, nofGen, # 
                                             stats=mstats, halloffame=hof, verbose=False, run=run, report_csv=report_csv, poolsize=poolsize, 
                                             datatrain=datatrain, datatest=datatest, target = settarget,poplnType =poplnType,FtEvlType =FtEvlType, pset =pset)#0.025
                    print(f'Taking stats for the last generation....Run: {run}')
                    #Collect stats for the last generation of each run.
                    lastgenstats(pop, toolbox, gen=nofGen, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest)#GEN....??  (9b)

            # elif poplnType == 'Features'  and TCMODE == 'APGP': # 
            #     poolsize = 25
            #     print(f'{TCMODE} on Features -------------!')               
            #     pop, log = APGPFtL(pop, toolbox, 0.9, 0.1, nofGen, # 
            #                              stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, poolsize=poolsize, datatrain=datatrain, datatest=datatest, target = settarget,poplnType =poplnType,FtEvlType =FtEvlType)#0.025
            # -----------------------------------------------------------------


        # ----------------------------------
        # ----------------------------------
        # APGP with Features and Adjusted R2
        # ----------------------------------
        if method == 'APGPAR2': # 
            print('doing --------- APGPAR2')
            for poolsize in THREADLIST:
                
                for i in range(1, runs+1):
                    run = i
                    pop = inipoplnF(popsize)
                    print(f'APGP-{poolsize} on Features -------------!')               
                    tag = f'APGP{poolsize}_{poplnType}_{FtEvlType}_{dataPLMB}'[:-4]#Standard GP - Steady State FtEvlType poplnType
                    report_csv = f"{reportfolder}PH6_{run_time}_{tag}.csv"

                    pop, log = APGPFtL(pop, toolbox, 0.9, 0.1, nofGen, # 
                                             stats=mstats, halloffame=hof, verbose=False, run=run, report_csv=report_csv, poolsize=poolsize, 
                                             datatrain=datatrain, datatest=datatest, target = settarget,poplnType =poplnType,FtEvlType =FtEvlType)#0.025
                    print(f'Taking stats for the last generation....Run: {run}')
                    #Collect stats for the last generation of each run.
                    lastgenstats(pop, toolbox, gen=nofGen, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest)#GEN....??  (9b)
        # ---------------------------------------------------------------------
 
            
if __name__ == "__main__":
    main()    



#==============================================================================
"""    
#==============================================================================
# -------------- Plot graphs of an individual and it's features ---------------
def plotind(xind):
    nodes, edges, labels = gp.graph(xind)
    str(xind)
    plt.figure(figsize=(10,7.5))
    g = nx.Graph(directed=True)
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.planar_layout(g)
    nx.draw_networkx_nodes(g, pos, node_size=600, node_color='#1f78b4')
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels, label_pos=10, font_size = 15)
    plt.show()

pop=population
for i in range(0,10): 
    print(f'{i} - length: {len(pop[i])}')
    plotind(pop[i])
    str(pop[i])
#==============================================================================
"""
#==============================================================================
"""
Select parents using double-tournament for bloat control.
counteval = 0
plotind(p1)
plotind(p2)
plotind(cand)
cxpb = 0.9
mutpb = 0.1
population = poplnFt(popsize)

run = 1
gen = 1
cxOver_db=[]

#  for i in range(0,len(pop)): print(f'{i} - length: {len(pop[i])} - {pop[i]}')

"""
#==============================================================================    
"""    
pop = poplnFt(15)
       
for i in range(0,len(pop)): 
    
for i in range(0,10): 
    print(f'{i} - length: {len(pop[i])}')
    print(str(pop[i]))
    
    
    plotind(pop[i])

individual = pop[3]
str(individual)

"""
#==============================================================================
"""
===============================================================================
pop = poplnFt(20)

# Plot to examine
for z in pop:
    plotind(z)

# select    
individual = pop[2]
--------------------------

for z in pop:
    error, evln_sec, error_test = evalfeat(z, datatrain, datatest)
    print()
    print(f'Training Error: {error}')
    print(f'Test     Error: {error_test}')
    print(f'Evaluation Time: {evln_sec}')

# check mutation --------
o = random.randint(1,9)
mutUniformFT(pop[o])
plotind(pop[o])

# check crossover -------
p = random.randint(0,9)
q = random.randint(0,9)

plotind(pop[p])
plotind(pop[q])
cxOnePointFt(pop[p], pop[q])
plotind(pop[p])
plotind(pop[q])

individual = pop[random.randint(0,len(pop)-1)]
===============================================================================
"""

