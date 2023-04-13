import numpy as np
import random
import time

def q_learning_epsilon_greedy(map, epsilon, location_index):
    step_count = 0
    q = np.zeros(shape = (100,4))
    current = (0,0)
    reward = map[current]
    alpha = 0.01
    beta = 0.9
    
    while reward != 1:
        reward = map[current]
        possible_actions = actions_check(map, current)
        current_index = location_index[current]
        next_location = (0,0)
        selected_action = []
        q_max = 0
        q_next = 0
        
        if len(possible_actions) > 0:
            #randomly exploit
            if random.uniform(0, 1) < epsilon:
                for action in possible_actions:
                    if q_next <= q[current_index][action]:
                        q_next = q[current_index][action]
                        selected_action = action
            #randomly explore
            else:
                selected_action = random.choice(possible_actions)

            next_location = next(current, selected_action) 
            next_location_index = location_index[next_location]
            next_location_possible_actions = actions_check(map, next_location)
           
            for action in next_location_possible_actions:
                if q_max <= q[next_location_index][action]:
                    q_max = q[next_location_index][action]

            #update Q model     
            q[current_index][action] = q[current_index][action] + alpha * (reward +(beta * q_max) - q[current_index][action])
            step_count += 1
            current = next_location
            
    return step_count, q

def actions_check(map, current):
    x = current[0]
    y = current[1]
    
    possible_actions = []
    if x-1 >= 0 and map[x-1][y]!=-100:
        possible_actions.append(0) #go up
    if x+1 < 10 and map[x+1][y]!=-100:
        possible_actions.append(1) #go down
    if y-1 >= 0 and map[x][y-1]!=-100:
        possible_actions.append(2) #go left
    if y+1 < 10 and map[x][y+1]!=-100:
        possible_actions.append(3) #go right
    
    return possible_actions

def next(current, action):
    x = current[0]
    y = current[1]

    #make sure the action is int
    if type(action) != int:
        action = action[0]

    if action == 0:
        x -= 1 #go up
    elif action == 1:
        x += 1 #go down
    elif action == 2:
        y -= 1 #go left
    elif action == 3:
        y += 1 #go right

    return (x,y)

#10*10 gridworld
map = np.zeros(shape=(10,10))
#wall location
wall = [(2,1),(2,2),(2,3),(2,4), (2,6), (2,7), (2,8), (3,4), (4,4), (5,4), (6,4), (7,4)]
#plus 1 reward location
P1 = [(5,5),]
#minus 1 reward location
M1 = [(3,3),(4,5),(4,6),(5,6),(5,8),(6,8),(7,3),(7,5),(7,6)]

#update the map
for i in wall:
    map[i] = -100
for i in P1:
    map[i] = 1
for i in M1:
    map[i] = -1

location_index =np.zeros(shape = (10,10), dtype = int)
n=0
for i in range(10):
    for j in range(10):
        location_index[i][j] = n
        n += 1

epsilon1 = 0.1
epsilon2 = 0.2
epsilon3 = 0.3
# print(map)
# print(epsilon)
# print(location_index)
num1, q1 = q_learning_epsilon_greedy(map, epsilon1, location_index)
num2, q2 = q_learning_epsilon_greedy(map, epsilon2, location_index)
num3, q3 = q_learning_epsilon_greedy(map, epsilon3, location_index)
print("epsilon = 0.1")
print("The iterations numbers: ",num1)
print("The Q values:")
print(q1)
print("\n")
print("epsilon = 0.2")
print("The iterations numbers: ",num2)
print("The Q values:")
print(q2)
print("\n")
print("epsilon = 0.3")
print("The iterations numbers: ",num3)
print("The Q values:")
print(q3)


