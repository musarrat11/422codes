import numpy as np

graph = []
with open('AlienData') as file:
    lines = file.readlines()
    rows = int(lines[0])
    cols = int(lines[1])
    lines = lines[2:]
for line in lines:
    graph.append(line.split())
# print(graph)
#rows = len(graph)
#cols = len(graph[0])
track = np.zeros((rows, cols))
visited = np.zeros((rows, cols))

for i in range(len(graph)):
    for j in range(len(graph[i])):
        if graph[i][j] == 'A':
            graph[i][j] = 1
            track[i][j] = 1
            visited[i][j] = 0
        elif graph[i][j] == 'H':
            graph[i][j] = 0
            visited[i][j] = 0
        else:
            graph[i][j] = 2
            visited[i][j] = 1


# print(visited)
# print(track)
def bfs(given, visitedd, parent):
    queue = []
    timetrack = []
    # row, col = start[0], start[1]
    # queue.append(start)
    # visitedd[row][col] = 1
    for row in range(len(parent)):
        for col in range(len(parent[row])):
            if parent[row][col] == 1:
                queue.append((row, col))
                visitedd[row][col] = 1
                time = 0
                while queue:
                    my_row, my_col = queue.pop(0)
                    # print(given[my_row][my_col])
                    timer = False
                    for k, l in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
                        if isvalid(visitedd, my_row + k, my_col + l):
                            visitedd[my_row + k][my_col + l] = 1
                            given[my_row + k][my_col + l] = 1
                            timer = True
                            queue.append((my_row + k, my_col + l))

                    if timer == True:
                        time += 1
                        # print(time)

                timetrack.append(time)
    return timetrack, given


def isvalid(visit, a, b):
    if a < 0 or b < 0 or a >= rows or b >= cols:
        return False
    if visit[a][b] == 1:
        return False
    else:
        return True


result = []
result, track = bfs(graph, visited, track)
print("time: ", max(result), "min")
count = 0
for row in range(len(track)):
    for col in range(len(track[row])):
        if track[row][col] == 0:
            count += 1
print(count, " survived")
