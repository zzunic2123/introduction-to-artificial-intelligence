import re
from itertools import combinations


def read_clauses_from_file(file_path):
    clauses_list = []
    with open(file_path, 'r') as file:
        for line in file:
            cleaned_line = line.strip().lower()
            if not cleaned_line.startswith('#') and cleaned_line:
                clauses_list.append(cleaned_line)
    return clauses_list


def read_user_commands_from_file(file_path):
    user_commands_list = []
    with open(file_path, 'r') as file:
        for line in file:
            cleaned_line = line.strip().lower()
            if not cleaned_line.startswith('#') and cleaned_line:
                cmd = cleaned_line[-1]
                clause_text = cleaned_line[:-2]
                user_commands_list.append((clause_text, cmd))
    return user_commands_list


def negate(str):
    if '~' in str:
        return re.sub('~', '', str)
    
    return '~' + str

def resolve(x,y):
    resolvents = set()
    x = to_set(x)
    y = to_set(y)
    for l1 in x:
        for l2 in y:
            if l1 == negate(l2) or l2 == negate(l1):
                cl = []
                for l in x.union(y):
                    if l not in (l1, l2):
                        if l in cl:
                            continue
                        if negate(l) not in cl:
                            cl.append(l)
                        else:
                            return ()
                        
                if cl:
                    resolvents.add(tuple(cl))
                else:
                    resolvents.add('')

    return resolvents

def to_set(y):
    set1 = set()

    if isinstance(y,tuple):
        for k in y:
            set1.add(k)
    else:
        set1.add(y)
    
    return set1

def to_str(x):
    if isinstance(x,tuple):
        str_elements = [str(element) for element in x]
        return " v ".join(str_elements)
    else:
        return str(x)




def select_clauses(clauses_set):
    selected = set()
    for x in clauses_set:
        for y in clauses_set:
            if x == y:
                continue
            set1 = to_set(y)
            for z in set1:
                if negate(z) in x:
                    selected.add((x,y))
    
    return selected

def pl_resolution(sos,clauses_set):
    new = set()
    memory = set()
    while True:
        for a,b in select_clauses(clauses_set):

            if a not in sos and b not in sos:
                 continue
            
            if tuple(a)+tuple(b) in memory:
                continue

            c = resolve(a,b)
        
            memory.add(tuple(a)+tuple(b))
            memory.add(tuple(b)+tuple(a))

            if not c:
                continue

            print( to_str(c) + ' <=== ' + to_str(b) + '  '+ to_str(a))
            
            if '' in c:
                return True
            else:
                new = new.union(c)

        if not new or new.issubset(clauses_set):
            return False
        
        sos = sos.union(new)
        clauses_set = clauses_set.union(new)
        
        sos =delete_unimportant(sos)
        clauses_set = delete_unimportant(clauses_set)
        new = delete_unimportant(new)

        new, sos,clauses_set = delete_redundant(new, sos,clauses_set)
        

def delete_unimportant(set1):
    temp = set()
    for x in set1:
        for y in set(x):
            if negate(y) in set(x):
                temp.add(x)
    
    for x in temp:
        if x in set1:
            set1.discard(x)

    return set1



def delete_redundant(new, sos,clauses):
    temp = set()
    all = new.union(sos)
    all = all.union(clauses)

    for x in all:
        for y in all:
            if set(x).issubset(set(y)) and len(x) != len(y):
                temp.add(y)

    
    for x in all:
        if len(set(x)) == 2:
            for y in set(x):
                if y in all:
                    temp.add(x)
    for x in temp:
        new.discard(x)
        sos.discard(x)
        clauses.discard(x)

    return new,sos,clauses


def print_res(clauses):
    
    clauses_set = set()
    sos = set()

    for x in clauses[:-1]:
        if 'v' in x:
            ntorka = tuple(x.split(' v '))
            flag = False

            for x in set(ntorka):
                for y in set(ntorka):
                    if negate(x) == y:
                        flag = True
            
            if flag:
                continue
            clauses_set.add(ntorka)
        else:
            clauses_set.add(x)

    if 'v' in clauses[-1]:
        for x in clauses[-1].split(' v '):
            print(x)
            clauses_set.add((negate(x)))
            sos.add(negate(x))
    else:
        clauses_set.add((negate(clauses[-1])))
        sos.add(negate(clauses[-1]))
    
    goal_clause = negate(clauses[-1])
    clauses[-1] = goal_clause

    for i,x in enumerate(clauses):
         print(str(i+1) + '. ' + x)

    print('===============')
    
    flag = pl_resolution(sos,clauses_set)


    if flag:
        print('===============')
        print('[CONCLUSION]: ' + negate(goal_clause) + ' is true')
    else:
       print('[CONCLUSION]: ' + negate(goal_clause) + ' is unknown') 

def isp(flag, goal):
    if flag:
        print('===============')
        print('[CONCLUSION]: ' + negate(goal) + ' is true')
    else:
       print('[CONCLUSION]: ' + negate(goal) + ' is unknown') 


def call_cooking(clauses, commands):
    clauses_set = set()
    sos = set()

    for x in clauses:
        if 'v' in x:
            ntorka = tuple(x.split(' v '))
            flag = False

            for x in set(ntorka):
                for y in set(ntorka):
                    if negate(x) == y:
                        flag = True
            
            if flag:
                continue
            clauses_set.add(ntorka)
        else:
            clauses_set.add(x)

    
    for x,c in commands:
        print('User command ' + negate(x) + ' ' + c)
        clauses_set_copy = clauses_set.copy()
        sos_copy = sos.copy()
        if c == '?':
            clauses_set_copy.add(negate(x))
            sos_copy.add(negate(x))
            # print()
            # print(clauses_set_copy)
            # print(sos_copy)
            # print()
            flag = pl_resolution(sos_copy,clauses_set_copy)
            isp(flag,negate(x)) 
        if c == '+':
            if 'v' in x:
                clauses_set.add(tuple(x.split(' v ')))
            else:
                clauses_set.add(x)
        if c == '-':
            if 'v' in x:
                clauses_set.discard(tuple(x.split(' v ')))
            else:
                clauses_set.discard(x)

    



import sys
def main():
    mode = sys.argv[1]
    clauses_file = sys.argv[2]
    if mode != "resolution":
        commands_file = sys.argv[3]

    clauses = read_clauses_from_file(clauses_file)

    if mode == "resolution":
        print_res(clauses)
    elif mode == "cooking":
        commands = read_user_commands_from_file(commands_file)
        call_cooking(clauses,commands)
    else:
        print("Invalid mode. Use either 'resolution' or 'cooking'.")

if __name__ == '__main__':
    main()
