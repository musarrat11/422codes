import random as r
import math

student_id = input("give your id: ")
turns = int(student_id[0]) * 2  # defender has one turn and attacker has one, total 2 #depth
defender_hp = int(student_id[-2:][::-1])
bullets = int(student_id[2])  # branchingFactor
HPrange = input("Minimum and Maximum value for the range of negative HP: ").split()
min = int(HPrange[0])
max = int(HPrange[1])
comparison_counter = 0
# leaf_nodes = [18, 13, 5, 12, 10, 5, 13, 7, 17, 8, 6, 8, 5, 11, 13, 18]
# leaf_nodes = [19, 22, 9, 2, 26, 16, 16, 27, 16]
leaf_nodes = []
for i in range(0, bullets ** turns):
    leaf_nodes.append(r.randrange(min, max + 1))
visited = [0 for i in range(len(leaf_nodes))]
length = len(leaf_nodes)


def alphaBeta(depth, node, maxturn, bulletsArray, alpha, beta):
    global visited
    if depth == 0:
        return bulletsArray[node]  # when depth is 0 we have reached the last level so it will backtrack
    elif maxturn:
        best_val = -math.inf  # alpha start
        for i in range(0, bullets):
            val = alphaBeta(depth - 1, node * bullets + i, False, bulletsArray, alpha, beta)
            # print("alpahanode", i,"value:", node * bullets + i)
            if best_val >= val:
                best_val = best_val
            else:
                best_val = val
            # best_val = max(best_val, val)
            if best_val >= alpha:
                alpha = best_val
            else:
                alpha = best_val
            # alpha = max(best_val, alpha)
            if beta <= alpha:
                visited[node * bullets + i] = 0
                # comparison_counter += 1
                break
        return best_val
    else:  # minturn
        best_val = math.inf
        for j in range(0, bullets):  # branchingfactor=2
            val = alphaBeta(depth - 1, node * bullets + j, True, bulletsArray, alpha, beta)
            print("betanode", node * bullets + j, "value:", j)
            visited[node * bullets + j] = 1
            if best_val >= val:
                best_val = val
            else:
                best_val = best_val
            # best_val = min(best_val, val)
            if best_val >= beta:
                beta = beta
            else:
                beta = best_val
            # beta = min(best_val, beta)
            if beta <= alpha:
                # visited[node * bullets + j] = 0
                # comparison_counter += 1
                # unvisited = unvisited + (bullets - (node * bullets + j))
                break
        return best_val


choseBullet = alphaBeta(turns, 0, True, leaf_nodes, -math.inf, math.inf)
print("1. Depth and Branches ratio is ", turns, ":", bullets)
print("2. Terminal States (leaf node values) are ", leaf_nodes)
print("3. Left life(HP) of the defender after maximum damage caused by the attacker is ", defender_hp - choseBullet)
count = 0
for i in range(len(visited)):
    if visited[i] == 0:
        count += 1
print("4. After Alpha-Beta Pruning Leaf Node Comparisons ", length - count)
