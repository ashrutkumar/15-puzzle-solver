# 15-puzzle-solver
 This is a 15-puzzle solver made for the course Artificial Intelligence(CS F407) of BITS Pilani, Goa Campus(Sem-I,2020-2021). 

 The A* graph search algorithm is used here with the heuristics Manhattan distance and Manhattan distance + Linear conflict.

 This solver only works for when the goal state has a blank/0 in the top-left corner i.e  
 [['0','1','2','3'],  
 ['4','5','6','7'],  
 ['8','9','A','B'],  
 ['C','D','E','F']]

 Various Optimizations are done to decrease the run-time some of which are:
 1. The puzzle is represented in the form of a string, rather than a list/tuple.
 2. Change in Heuristic is calculated from the parent state, rather than calculating the heuristic from scratch for each state.
 
 The desciption of the optimizations, optimality of the algorithm and the results of the solver can be found in the Results.pdf .
