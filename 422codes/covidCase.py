import numpy as np

graph = []
with open('covidData') as file:
    lines = file.readlines()
for line in lines:
    graph.append(line.split())

rows = len(graph)
cols = len(graph[0])
visited = np.zeros((rows, cols))

for i in range(len(graph)):
    for j in range(len(graph[i])):
        if graph[i][j] == 'Y':
            graph[i][j] = 1
            visited[i][j] = 0
        else:
            graph[i][j] = 0
            visited[i][j] = 1

#print(graph)
#print(visited)


def isvalid(row, col):
    if row < 0 or col < 0 or row >= rows or col >= cols:
        return False
    if visited[row][col] == 1:
        return False
    return True


def visitDFS(time, visited, row, col):
    visited[row][col] = 1
    time=1
    for i in range(row - 1, row + 2, 1):
        for j in range(col - 1, col + 2, 1):
            if isvalid(i, j):
                time += visitDFS(time,visited, i, j)
                #time += 1
                #print(time)
                #visitDFS(time,visited, i, j)
    #print("bla",time)
    return time


area = []
for row in range(rows):
    for col in range(cols):
        if graph[row][col] == 1 and visited[row][col] == 0:
            time = 0
            size=visitDFS(time, visited, row, col)
            area.append(size)
print(max(area))
