# Applications of Time Control


This code set of code is used assess the evaluation time schemes (introduced in the publications listed beolow) on a variety of GP applications. The problem domains are as follows:

## Boolean logic - Multiplexer
For the Boolean logic applications, the multiplexer and even-parity problems are used. The challenge in the multiplexer problem (henceforth termed Multiplexer) \cite{oai:IEEE-CS:10.1109/EH.1999.785434, koza1992genetic} is for GP to use Boolean logic operators to reproduce the behaviour of an electronic multiplexer given all its possible input and output values. 

## Boolean logic - Even Parity
The challenge in the even-parity problem (Parity) \cite{harding2009self, koza1992genetic} is for GP to use Boolean logic operators to find a solution that produces the value of the Boolean even parity given $n$ independent Boolean inputs.  

## Robot Rontrol: ANT
For the robot control application, GP is used to evolve solutions to the well known artificial ant problem (ANT) \cite{iba:1996:aigp2, koza1992genetic}. The challenge in ANT is to evolve a routine for navigating a robotic ant on a virtual field to help it find food items within a given time limit. 

## Classification with GPML
The classification application uses a hybridisation of GP with machine learning (GPML). Like MLR-GP in Chapter~\ref{Chapter8}, the hybridisation involves GP and machine learning (GPML), whereas here it is for classification; GP engineers a set of features and then logistic regression \cite{kleinbaum2002logistic} uses these features to build a classification model. 



# Derived Publications:

Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah.  *“Evolving Simple and Accurate Symbolic Regression Models via Asynchronous Parallel Computing"* In: Applied Soft Computing 104 (2021), p. 107198. ISSN: 1568-4946.
 URL: https://doi.org/10.1016/j.asoc.2021.107198

Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah.  *“Time control or size control? reducing complexity and improving the accuracy of genetic programming models"*, In: European Conference on Genetic Programming, Springer, 2020, pp. 195–210. URL: https://doi.org/10.1007/978-3-030-44094-7_13

Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah. *“Leveraging asynchronous parallel computing to produce simple genetic programming computational models",* In: Proceedings of the 35th Annual ACM Symposium on Applied Computing, SAC ’20, Association for Computing Machinery, NY, USA, 2020, p521–528. URL: https://doi.org/10.1145/3341105.3373921

Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah. *“Feature Engineering for Enhanced Performance of Genetic Programming Models",*
In: GECCO '20 Companion, Genetic and Evolutionary Computation Conference Companion, July 2020. URL: https://doi.org/10.1145/3377929.3390078.
 
Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah. *“Improving the Generalisation of Genetic Programming Models with Evaluation Time and Asynchronous Parallel Computing"*, In: GECCO '21 Companion, Genetic and Evolutionary Computation Conference Companion, July 2021. URL: https://doi.org/10.1145/3449726.3459583

