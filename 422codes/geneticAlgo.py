import random as r
import numpy as np

model = []
with open('geneticData') as file:
    lines = file.readlines()
    transactions = int(lines[0])
    lines = lines[1:]
for line in lines:
    a = line.split()
    b = a[0]
    c = a[1]
    model.append((b, c))


# print(model)


def fitnessCalculate(population):
    total_sum = []
    for p in population:
        sum = 0
        for i in range(transactions):
            if p[i] == "1":
                status, value = model[i]
                if status == "l":
                    sum -= int(value)
                else:
                    sum += int(value)
        total_sum.append(sum)
    # print(total_sum.index(0))
    # print(total_sum)

    # print(total_sum.count(0))

    for i in range(len(total_sum)):
        b = False
        if total_sum[i] == 0 and int(population[i]) != 0:
            b = True
            break
            #print(population[i])
            # print(population[total_sum.index(0)])
        else:
            b = False
    if b == True:
        print(population[i])
    else:
        print(-1)


# step 1: populate the data
def generatePopulation():
    population_size = 2 ** transactions - 1
    p = 1
    population = []
    while p <= population_size:
        popul = ""
        for i in range(transactions):
            pu = str(r.randrange(0, 2))
            popul += pu
        if popul not in population:
            population.append(popul)
            p += 1
    # print(population.count('1011010'))
    fitnessCalculate(population)


# step 2:calculate fitness which in this case is 0
# step 3:crossover
# step 4:mutation
generatePopulation()
