import pprint
import heapq


state_space_dict = dict()
heuristic_dict = dict()


def read_state_space_file(path):
    init_state, goal_state = None, None

    lines= open(path, 'r').readlines()

    for line in lines:
        # Split the line into place names and distances
        if ':' in line and '#' not in line:
            parts = line.split(':')
            name = parts[0]
            if len(parts) > 1:
                distances = {}
                pairs = parts[1].split()

                for pair in pairs:
                    place, distance = pair.split(',')
                    distances[place] = int(distance)
            else:
                distances = {}
            state_space_dict[name] = distances
        elif '#' not in line:
            if not init_state:
                init_state = line
            elif not goal_state:
                goal_state = line
    
    return init_state.strip(), goal_state.strip()

def read_heuristic_file(path):
    lines= open(path, 'r').readlines()

    for line in lines:
        if '#' not in line:
            place, value = line.strip().split(': ')
            heuristic_dict[place] = int(value)

#succ child nodes
def bfs(init,goal):
    queue = [init]
    states_visited = set()
    path = {}
    while queue:
        n = queue.pop(0)
        states_visited.add(n)
        #print(queue)
        if n in goal:
            return True, len(states_visited),path,n
        
       # sortirani = sorted(list(state_space_dict[n].keys()))

        for x in state_space_dict[n]:
            if x not in states_visited:
                    path[x] = n
                    queue.append(x)
    return False,len(states_visited),path,n


def print_bfs(init, goal):

    result, states_visited,path,n = bfs(init,goal)
    # print(result)
    # print(states_visited)
    # print(path)
    if result:
        result = 'yes'
    else:
        result = 'no'

    path_cost = 0
    new = n
    path_list = []
    while 1:
        path_list.append(new)
        if new == init:
            break
        curr = path[new]
        path_cost += state_space_dict[curr][new]
        new = curr

    path_length = len(path_list)
    final_path = ' => '.join(path_list[::-1]) 

    print(
        '# BFS' + '\n'
        '[FOUND_SOLUTION]: ' + str(result) + '\n'
          '[STATES_VISITED]: ' + str(states_visited) + '\n'
          '[PATH_LENGTH]: ' + str(path_length) + '\n'
          '[TOTAL_COST]: ' + str(float(path_cost)) + '\n'
          '[PATH]: ' + final_path
          )

def ucs(init, goal):
    states_visited = set()
    #priority_queue = [(0,init,[])]
    priority_queue = [(0,init)]
    path_dict = {}

    while priority_queue:
        #print(priority_queue)
        #cost,n,path = heapq.heappop(priority_queue)
        cost,n = heapq.heappop(priority_queue)
        #path = path + [n]
        states_visited.add(n)

        if n in goal:
            return True,len(states_visited),cost,path_dict,n
        
        for x in state_space_dict[n]:
            if x not in states_visited:
                path_dict[x] = n
                heapq.heappush(priority_queue,(cost + state_space_dict[n][x],x))

    return False,len(states_visited),cost,path_dict,n

def print_ucs(init, goal):

    result, states_visited, cost, path,n = ucs(init_state, goal_state)
    
    if result:
        result = 'yes'
    else:
        result = 'no'

    new = n
    path_list = []
    while 1:
        path_list.append(new)
        if new == init:
            break
        curr = path[new]
        new = curr

    path_length = len(path_list)
    final_path = ' => '.join(path_list[::-1]) 

    print(
        '# UCS' + '\n'
        '[FOUND_SOLUTION]: ' + str(result) + '\n'
          '[STATES_VISITED]: ' + str(states_visited) + '\n'
          '[PATH_LENGTH]: ' + str(path_length) + '\n'
          '[TOTAL_COST]: ' + str(float(cost)) + '\n'
          '[PATH]: ' + final_path
          )

def a(init, goal):
    states_visited = set()
    #priority_queue = [(0,init,[])]
    priority_queue = [(0,0,init)]
    path_dict = {}

    while priority_queue:
        #cost,n,path = heapq.heappop(priority_queue)
        total,cost,n = heapq.heappop(priority_queue)
        #path = path + [n]
        states_visited.add(n)

        if n in goal:
            return True,len(states_visited),cost,path_dict,n
        
        for x in state_space_dict[n]:
            if x not in states_visited:
                path_dict[x] = n
                heapq.heappush(priority_queue,(cost + state_space_dict[n][x] + heuristic_dict[x],cost + state_space_dict[n][x],x))
            elif x in priority_queue:
                c = -1
                for i in range(len(priority_queue)):
                    t,c,n = priority_queue[i]
                    if n == x:
                        break
                if(c > cost + state_space_dict[n][x]):
                    heapq.heappush(priority_queue,(cost + state_space_dict[n][x] + heuristic_dict[x],cost + state_space_dict[n][x],x))


    return False,len(states_visited),cost,path_dict,n

def print_a(init, goal,file_path):

    result, states_visited, cost, path,n = a(init_state, goal_state)

    if result:
        result = 'yes'
    else:
        result = 'no'

    new = n
    path_list = []
    while 1:
        path_list.append(new)
        if new == init:
            break
        curr = path[new]
        new = curr

    path_length = len(path_list)
    final_path = ' => '.join(path_list[::-1]) 

    print(
        '# A-STAR ' + file_path + '\n'
        '[FOUND_SOLUTION]: ' + str(result) + '\n'
          '[STATES_VISITED]: ' + str(states_visited) + '\n'
          '[PATH_LENGTH]: ' + str(path_length) + '\n'
          '[TOTAL_COST]: ' + str(float(cost)) + '\n'
          '[PATH]: ' + final_path
          )
    

def is_optimistic(goal, path):

    print('# HEURISTIC-OPTIMISTIC '+ path)
    flag = True
    for x in state_space_dict:
        k,y,cost,m,n = ucs(x,goal)
        if cost < heuristic_dict[x]:
            flag = False
            print('[CONDITION]: [ERR] h(' + x + ') <= h*: '+ str(float(heuristic_dict[x])) +' <= '+str(float(cost)))
        else:
            print('[CONDITION]: [OK] h(' + x + ') <= h*: '+ str(float(heuristic_dict[x])) +' <= '+str(float(cost)))
    
    if not flag:
        print('[CONCLUSION]: Heuristic is not optimistic.')
    else:
        print('[CONCLUSION]: Heuristic is optimistic.')

    print()

def is_consistent(path):
    
    print("# HEURISTIC-CONSISTENT " + path)

    flag = True
    for main in state_space_dict:

        for x in state_space_dict[main]:
#[CONDITION]: [OK] h(Baderna) <= h(Kanfanar) + c: 25.0 <= 30.0 + 19.0
            if float(heuristic_dict[main]) <= float(heuristic_dict[x]) + float(state_space_dict[main][x]):
                print('[CONDITION]: [OK] h(' + main + ') <= h(' + x + ') + c: ' + str(float(heuristic_dict[main])) + ' <= ' + str(float(heuristic_dict[x])) + ' + ' + str(float(state_space_dict[main][x])))
            else:
                flag = False
                print('[CONDITION]: [ERR] h(' + main + ') <= h(' + x + ') + c: ' + str(float(heuristic_dict[main])) + ' <= ' + str(float(heuristic_dict[x])) + ' + ' + str(float(state_space_dict[main][x])))

    if not flag:
        print('[CONCLUSION]: Heuristic is not consistent.')
    else:
        print('[CONCLUSION]: Heuristic is consistent.')

    print()


#def is_consistent()
import sys

# >>> python solution.py --alg astar --ss istra.txt --h istra_heuristic.txt

algorithm,state_space_path,heuristic_path = None, None,None
check_optimistic, check_consistent= False, False

for i in range(1,len(sys.argv)):
    if sys.argv[i] == '--alg':
        algorithm = sys.argv[i+1]
    if sys.argv[i] == '--ss':
        state_space_path = sys.argv[i+1]
    if sys.argv[i] == '--h':
        heuristic_path = sys.argv[i+1]
    if sys.argv[i] == '--check-optimistic':
        check_optimistic = True
    if sys.argv[i] == '--check-consistent':
        check_consistent = True

if state_space_path:
    init_state, goal_state = read_state_space_file(state_space_path)
    if heuristic_path:
        read_heuristic_file(heuristic_path)

if algorithm:
    if algorithm == 'bfs':
        print_bfs(init_state,goal_state)
    elif algorithm == 'ucs':
        print_ucs(init_state,goal_state)
    elif algorithm == 'astar':
        print_a(init_state,goal_state,heuristic_path)

if check_optimistic:
    is_optimistic(goal_state,heuristic_path)

if check_consistent:
    is_consistent(heuristic_path)






