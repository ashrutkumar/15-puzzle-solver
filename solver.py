#!/usr/bin/env python3
import time
from heapq import * 
import numpy as np 
import sys



def process_path(path :str) -> list:
    """ 
    Utility function to convert path into proper format for printing.
    Accepts a string containing the encoded representation of steps for efficiency.
    Returns a list of strings containing the order of steps taken to solve the puzzle.
    """
    conversion_dict={'u':"Up",'l':"Left",'r':"Right",'d':"Down"}
    
    return [conversion_dict[i] for i in path]

def get_neighbours(state: str,path: str) -> list:
    """
    Utility function to get the nodes formed after expanding current node by doing valid steps as per the puzzle.
    Accepts current state and current path history as parameters.
    Return a list of tuple containig states ,their corresponding path history and the change in heuristic value from the parent(delta).
    """
    pos_zero = state.index('0') #finding position of zero
    i,j = pos_zero//4, pos_zero%4
    i_4 = 4*i

    correct_pos={'0': [0, 0], '1': [0, 1], '2': [0, 2], '3': [0, 3], '4': [1, 0], '5': [1, 1], '6': [1, 2],
                 '7': [1, 3], '8': [2, 0], '9': [2, 1], 'A': [2, 2], 'B': [2, 3], 'C': [3, 0], 'D': [3, 1],
                 'E': [3, 2], 'F': [3, 3]}

    #calculating moves possible based on state
    possible_moves={'u','l','r','d'}
    
    if i ==0 : possible_moves.remove('u')
    if j ==0 : possible_moves.remove('l')
    if i ==3 : possible_moves.remove('d')
    if j ==3 : possible_moves.remove('r')

    opposite_dict = {'u':'d','l':'r','r':'l','d':'u'}
    #removing the move opposite to last path as it will take it back to last state, which will never result in optimal solution
    if len(path)!=0: 
        opposite_move = opposite_dict[path[-1]] 
        if opposite_move in possible_moves:
            possible_moves.remove(opposite_move)

    neighbours = []
    
    for move in possible_moves:

        if move == 'u':
            replacee = state[i_4 + j - 4] #the tile which comes in the position of 0
            new_state = state[:i_4 +j - 4]+'0'+state[i_4+j-3:i_4+j]+replacee+state[i_4+j+1:] #calculating new state after the move
            actual_i,actual_j = correct_pos[replacee]
            #calculating the change in manhattan distance by the move
            delta_md = (abs(actual_i - (i) )+ abs(actual_j - j)) - (abs(actual_i - (i-1) )+ abs(actual_j - j)) 
            #calculating the change in linear conflict by the move
            delta_lc = 0
            if actual_i == i: #if replacee belongs in new row, then lc might increase
                for itr in range(4):
                    if itr==j:continue
                    temp = tuple(correct_pos[state[4*i + itr]])
                    if temp[0] == i:
                        if itr > j and temp[1]<actual_j : delta_lc+=1
                        if itr < j and temp[1]>actual_j : delta_lc+=1
            if actual_i ==i-1: #if replacee belonged to old row, then lc might decrease
                for itr in range(4):
                    if itr==j:continue
                    temp = tuple(correct_pos[state[4*(i-1) + itr]])
                    if temp[0] == i-1:
                        if itr > j and temp[1]<actual_j : delta_lc-=1
                        if itr < j and temp[1]>actual_j : delta_lc-=1
                        
            delta = delta_md + 2*delta_lc # the total change in heuristic score by this move
            neighbours.append([new_state,path + 'u',delta])
            
            
        elif move == 'l':
            replacee = state[i_4 + j - 1] #the tile which comes in the position of 0
            new_state= state[:i_4+j-1]+'0'+replacee+state[i_4+j+1:] #calculating new state after the move
            actual_i,actual_j = correct_pos[replacee]
            #calculating the change in manhattan distance by the move
            delta_md = (abs(actual_i - (i) )+ abs(actual_j - j)) - (abs(actual_i - i )+ abs(actual_j - (j-1)))
            #calculating the change in linear conflict by the move
            delta_lc = 0
            if actual_j == j: #if replacee belongs in new column, then lc might increase
                for itr in range(4):
                    if itr==i:continue
                    temp = tuple(correct_pos[state[4*itr + j]])
                    if temp[1] == j:
                        if itr > i and temp[0]<actual_i : delta_lc+=1
                        if itr < i and temp[0]>actual_i : delta_lc+=1
            if actual_j ==j-1: #if replacee belonged to old column, then lc might decrease
                for itr in range(4):
                    if itr==i:continue
                    temp = tuple(correct_pos[state[4*itr + j - 1]])
                    if temp[1] == j-1:
                        if itr > i and temp[0]<actual_i : delta_lc-=1
                        if itr < i and temp[0]>actual_i : delta_lc-=1      
            delta = delta_md + 2*delta_lc # the total change in heuristic score by this move
            neighbours.append([new_state,path + 'l',delta])

        elif move == 'd':
            replacee = state[i_4 + j + 4] #the tile which comes in the position of 0
            new_state = state[:i_4 +j] + replacee+ state[i_4+j+1:i_4+j+4]+'0'+state[i_4+j+5:] #calculating new state after the move
            actual_i,actual_j = correct_pos[replacee]
            #calculating the change in manhattan distance by the move
            delta_md = (abs(actual_i - (i) )+ abs(actual_j - j)) - (abs(actual_i - (i+1) )+ abs(actual_j - j))
            #calculating the change in linear conflict by the move
            delta_lc = 0
            if actual_i == i: #if replacee belongs in new row, then lc might increase
                for itr in range(4):
                    if itr==j:continue
                    temp = tuple(correct_pos[state[4*i + itr]])
                    if temp[0] == i:
                        if itr > j and temp[1]<actual_j : delta_lc+=1
                        if itr < j and temp[1]>actual_j : delta_lc+=1
            if actual_i ==i+1: #if replacee belonged to old row, then lc might decrease
                for itr in range(4):
                    if itr==j:continue
                    temp = tuple(correct_pos[state[4*(i+1) + itr]])
                    if temp[0] == i+1:
                        if itr > j and temp[1]<actual_j : delta_lc-=1
                        if itr < j and temp[1]>actual_j : delta_lc-=1

            delta = delta_md + 2*delta_lc # the total change in heuristic score by this move
            neighbours.append([new_state,path + 'd',delta])

        elif move == 'r':
            replacee = state[i_4 + j + 1] #the tile which comes in the position of 0
            new_state = state[:i_4+j]+replacee+'0'+state[i_4+j+2:] #calculating new state after the move
            actual_i,actual_j = correct_pos[replacee]
            #calculating the change in manhattan distance by the move
            delta_md = (abs(actual_i - (i) )+ abs(actual_j - j)) - (abs(actual_i - i )+ abs(actual_j - (j+1)))
            #calculating the change in linear conflict by the move
            delta_lc = 0
            if actual_j == j: #if replacee belongs in new column, then lc might increase
                for itr in range(4):
                    if itr==i:continue
                    temp = tuple(correct_pos[state[4*itr + j]])
                    if temp[1] == j:
                        if itr > i and temp[0]<actual_i : delta_lc+=1
                        if itr < i and temp[0]>actual_i : delta_lc+=1
            if actual_j ==j+1: #if replacee belonged to old column, then lc might decrease
                for itr in range(4):
                    if itr==i:continue
                    temp = tuple(correct_pos[state[4*itr + j + 1]])
                    if temp[1] == j+1:
                        if itr > i and temp[0]<actual_i : delta_lc-=1
                        if itr < i and temp[0]>actual_i : delta_lc-=1

            delta = delta_md + 2*delta_lc # the total change in heuristic score by this move
            neighbours.append([new_state,path + 'r',delta])


        # comment above for loop and uncomment  below for optimised manhattan heuristic
        # for move in possible_moves:
        #     if move == 'u':
        #         replacee = state[i_4 + j - 4]
        #         new_state = state[:i_4 +j - 4]+'0'+state[i_4+j-3:i_4+j]+replacee+state[i_4+j+1:]
        #         actual_i,actual_j = correct_pos[replacee]
        #         delta = (abs(actual_i - (i) )+ abs(actual_j - j)) - (abs(actual_i - (i-1) )+ abs(actual_j - j))
        #         neighbours.append([new_state,path + 'u',delta])       
        #     elif move == 'l':
        #         replacee = state[i_4 + j - 1]
        #         new_state= state[:i_4+j-1]+'0'+replacee+state[i_4+j+1:]
        #         actual_i,actual_j = correct_pos[replacee]
        #         delta = (abs(actual_i - (i) )+ abs(actual_j - j)) - (abs(actual_i - i )+ abs(actual_j - (j-1)))
        #         neighbours.append([new_state,path + 'l',delta])
        #     elif move == 'd':
        #         replacee = state[i_4 + j + 4]
        #         new_state = state[:i_4 +j] + replacee+ state[i_4+j+1:i_4+j+4]+'0'+state[i_4+j+5:]
        #         actual_i,actual_j = correct_pos[replacee]
        #         delta = (abs(actual_i - (i) )+ abs(actual_j - j)) - (abs(actual_i - (i+1) )+ abs(actual_j - j))
        #         neighbours.append([new_state,path + 'd',delta])
        #     elif move == 'r':
        #         replacee = state[i_4 + j + 1]
        #         new_state = state[:i_4+j]+replacee+'0'+state[i_4+j+2:]
        #         actual_i,actual_j = correct_pos[replacee]
        #         delta = (abs(actual_i - (i) )+ abs(actual_j - j)) - (abs(actual_i - i )+ abs(actual_j - (j+1)))
        #         neighbours.append([new_state,path + 'r',delta])

        #comment above for loops and uncomment below for normal manhattan / manhattan with linear conflict heuristic
        # for move in possible_moves:
        #     if move == 'u':
        #         replacee = state[i_4 + j - 4]
        #         new_state = state[:i_4 +j - 4]+'0'+state[i_4+j-3:i_4+j]+replacee+state[i_4+j+1:]
        #         neighbours.append([new_state,path + 'u'])                
        #     elif move == 'l':
        #         replacee = state[i_4 + j - 1]
        #         new_state= state[:i_4+j-1]+'0'+replacee+state[i_4+j+1:]
        #         neighbours.append([new_state,path + 'l'])
        #     elif move == 'd':
        #         replacee = state[i_4 + j + 4]
        #         new_state = state[:i_4 +j] + replacee+ state[i_4+j+1:i_4+j+4]+'0'+state[i_4+j+5:]
        #         neighbours.append([new_state,path + 'd'])
        #     elif move == 'r':
        #         replacee = state[i_4 + j + 1]
        #         new_state = state[:i_4+j]+replacee+'0'+state[i_4+j+2:]
        #         neighbours.append([new_state,path + 'r'])

    
    return neighbours

def heuristic_score(state: str) -> int:
    """
    Utility Function to calculate the expected cost to finish the puzzle from the current state using various heuristics.
    Accepts the current state as a parameter.
    Return an integer corresponding to the expected cost.
    """

    correct_pos={'0': [0, 0], '1': [0, 1], '2': [0, 2], '3': [0, 3], '4': [1, 0], '5': [1, 1], '6': [1, 2],
                 '7': [1, 3], '8': [2, 0], '9': [2, 1], 'A': [2, 2], 'B': [2, 3], 'C': [3, 0], 'D': [3, 1],
                 'E': [3, 2], 'F': [3, 3]}
    #calculating manhattan distance
    manhattan=0
    for i,j in enumerate(state):
        actual_i,actual_j = correct_pos[j]
        if j == '0': continue
        manhattan+=(abs(actual_i-i//4)+abs(actual_j-i%4))  #

    #return manhattan # uncomment for manhattan and manhattan optimised heuristic and comment rest below

    #reference for linear conlict code : https://github.com/Subangkar/N-Puzzle-Problem-CPP-Implementation-using-A-Star-Search

    #calculating the number of linear conflicts i.e. two tiles that are in the same row or column,
    #and their goals are also in the same row or column, but they're in the wrong order.

    #vertical linear conflicts
    lin_conflict = 0
    for i in range(0,4):
        for j in range(0,4):
            for k in range(j+1,4):
                if state[4*i+j]!='0' and state[4*i+k]!='0':
                     if i == correct_pos[state[4*i+j]][0]  == correct_pos[state[4*i+k]][0] :
                         if correct_pos[state[4*i+j]][1] > correct_pos[state[4*i+k]][1]:
                             lin_conflict+=1

    #horizontal linear conflicts
    for i in range(0,4):
        for j in range(0,4):
            for k in range(j+1,4):
                if state[4*j+i]!='0' and state[4*k+i]!='0':
                    if i == correct_pos[state[4*j+i]][1]  == correct_pos[state[4*k+i]][1]:
                        if correct_pos[state[4*j+i]][0] > correct_pos[state[4*k+i]][0]:
                            lin_conflict+=1
                             
    return manhattan + 2*lin_conflict

    #returns heuristic value of current state




def FindMinimumPath(initialState,goalState):
    nodesGenerated=0 # This variable should contain the number of nodes that were generated while finding the optimal solution
    
    init_state_str = ''.join(np.array(initialState).flatten()) #converting to string for efficiency
    goal_state_str = ''.join(np.array(goalState).flatten()) #converting to string for efficient equality check

 

    queue = [] # the frontier set / priority queue
    visited = set() # the visited set
    initial_path = '' #initializing the path
    
    heappush(queue,[heuristic_score(init_state_str),init_state_str,initial_path]) # pushing the node in frontier set
    nodesGenerated+=1

    while queue:
        lowest_score,current_state,current_path = heappop(queue) #popping the node with minimum cost till now

        visited.add(current_state) #adding to visited state    

        if current_state==goal_state_str:    #check for goal state
            return process_path(current_path),nodesGenerated #returning path and number of nodes

        neighbour_nodes = get_neighbours(current_state,current_path)  #expanding the current node
        
        for temp_state,temp_path,delta in neighbour_nodes:  #adding expanded nodes to frontier set        
            if temp_state not in visited:
                # calculating f(n+1) = g(n+1) + h(n+1) = g(n) + 1 + h(n) + (h(n+1)-h(n)) = g(n+1) + h(n+1)
                # g(n+1) = g(n) + 1 as step cost is always 1
                # delta = h(n+1) - h(n)
                expected_cost = lowest_score + 1 + delta 
                heappush(queue,[expected_cost,temp_state,temp_path]) #pushing node in frontier set
                nodesGenerated+=1            
        # comment above for loop and uncomment below for normal manhattan/ manhattan with linear conflict heuristics
        # for temp_state,temp_path in neighbour_nodes:  #adding expanded nodes to frontier set
        #     if temp_state not in visited:
        #         expected_cost = heuristic_score(temp_state) + len(temp_path) # calculating f(n) = g(n) + h(n)
        #         heappush(queue,[expected_cost,temp_state,temp_path]) #pushing node in frontier set
        #         nodesGenerated+=1    
    
    return ["Not solved"],nodesGenerated  #will reach here only if goal state not found, i.e puzzle is not solvable

def ShowState(state,heading=''):
    print(heading)
    for row in state:
        print(*row, sep = " ")

def verify_util(state,moves,goalState):
    state = np.array(state)
    i,j = np.where(state == '0')
    i,j = i[0],j[0]
    for move in moves:
        i,j = np.where(state == '0')
        i,j = i[0],j[0]
        if move =="Up":
            state[i][j], state[i-1][j] = state[i-1][j],state[i][j]
        if move =="Down":
            state[i][j], state[i+1][j] = state[i+1][j],state[i][j]
        if move =="Left":
            state[i][j], state[i][j-1] = state[i][j-1],state[i][j]
        if move =="Right":
            state[i][j], state[i][j+1] = state[i][j+1],state[i][j]
    ShowState(state,"Puzzle after the moves is :")
    if list([list(i) for i in state]) != goalState :
        print("Solver failed :/") 

def main():
    if len(sys.argv)<2:
        itr = ['1','2','3','4']
    else :
        itr = sys.argv[1:]
    for i in itr:
        print("File Number : ",i)
        with open("initial_state"+i+".txt", "r") as file: 
            initialState = [[x for x in line.split()] for i,line in enumerate(file) if i<4]
        ShowState(initialState,'Initial state:')
        goalState = [['0','1','2','3'],['4','5','6','7'],['8','9','A','B'],['C','D','E','F']]
        start = time.time()
        minimumPath, nodesGenerated = FindMinimumPath(initialState,goalState)
        timeTaken = time.time() - start        
        print('Output:')
        print('   Minimum path cost : {0}'.format(len(minimumPath)))
        print('   Actions in minimum path : {0}'.format(minimumPath))
        print('   Nodes generated : {0}'.format(nodesGenerated))
        print('   Time taken : {0} s'.format(round(timeTaken,4)))
        verify_util(initialState,minimumPath,goalState)

if __name__=='__main__':
    main()