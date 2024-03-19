import numpy as np
import matplotlib.pyplot as plt
import random


# Objective Function
def objective(x1, x2):
    return 8 - (x1 + 0.0317) ** 2 + x2 ** 2

# Objective function with constraint
def penaltyObjective(x1, x2):
    return (8 - (x1 + 0.0317) ** 2 + x2 ** 2) - abs((x1 + x2 - 1))


# This function splits the solution into two parts
# And decode it into integers and return two integer numbers
def binaryToInteger(part1, part2):
    # Two parts with the same length
    x_min = -2
    x_max = 2
    sum1 = 0
    sum2 = 0
    for i in range(len(part1)):
        sum1 += part1[i] * (2 ** (len(part1) - i - 1))
        sum2 += part2[i] * (2 ** (len(part2) - i - 1))
    x1 = round(x_min + (sum1 / (2 ** len(part1) - 1)) * (x_max - x_min), 4)
    x2 = round(x_min + (sum2 / (2 ** len(part2) - 1)) * (x_max - x_min), 4)
    return x1, x2

def graycodeToBinary(part1 , part2):
    # Two parts are the same length
    length = len(part1)
    binary1 = np.zeros(length , dtype=int)
    binary1[0] = part1[0]
    binary2 = np.zeros(length,dtype=int)
    binary2[0] = part2[0]
    for i in range(1,length):
        binary1[i] = binary1[i-1] ^ part1[i] #XOR Operation to convert gray into binary
        binary2[i] = binary2[i-1] ^ part2[i]

    return binary1, binary2

def grayToInteger(part1 , part2):
    binary1 , binary2 = graycodeToBinary(part1 , part2)
    return binaryToInteger(binary1 , binary2)

# Calculate Fitness of each solution by substitute values of x and y in objective function
def calculateFitness(pop):
    size = len(pop)
    Fitness = np.zeros(size, dtype=float)
    for i in range(size):
        # Split the solution into two parts and decode them
        part1 = pop[i][:5]
        part2 = pop[i][5:]
        x1, x2 = binaryToInteger(part1, part2)
        Fitness[i] = objective(x1, x2)

    return Fitness


# This function computes the probability and prevents zero-sum.
def adjustedProbabilities(Fitness):
    minFitness = np.min(Fitness)
    adjustedFitness = Fitness - minFitness + 1
    probability = adjustedFitness / np.sum(adjustedFitness)
    return probability



# Selection using roulette wheel technique
def selection(pop, Fitness):
    # This prevents the probability of the total fitness be equal zero
    probability = adjustedProbabilities(Fitness)

    # Compute the cumulative probability of the fitness
    cumulative = np.cumsum(probability)
    i = 0
    j = 0
    # Get two distinct parents based on their probabilities
    while i == j:
        i = 0
        j = 0
        num1 = random.random()
        num2 = random.random()
        while num1 > cumulative[i]:
            i += 1
        while num2 > cumulative[j]:
            j += 1
    return pop[i] , pop[j]


# Recombination operation using one-point crossover
def crossover(p1, p2, pCross):
    num = random.random()
    if num < pCross:
        # Take the index of separation randomly
        split_point = random.randint(1, len(p1) - 1)
        firstChild = np.concatenate((p1[:split_point] , p2[split_point:]))
        secondChild = np.concatenate((p2[:split_point] , p1[split_point:]))
        return firstChild, secondChild
    else:
        return p1, p2



# Mutation Operation using bit-flip mutation way
def mutation(pop, pMutation):
    for i in range(len(pop)):
        num = random.random()
        if num < pMutation:  # mutate this solution
            randomIndex = random.randint(0, len(pop[0]) - 1)
            if pop[i][randomIndex] == 0:
                pop[i][randomIndex] = 1
            else:
                pop[i][randomIndex] = 0
        else:  # No Mutation, then continue
            continue


# Genetic Algorithm function.
def geneticAlgorithm(Population, generations, chromosomeSize, pCross, pMutation):
    populationSize = len(Population)
    newGenerations = np.zeros((populationSize, chromosomeSize) ,dtype=int)
    bestFitness = np.zeros(generations)

    for i in range(generations):
        # Calculate the fitness
        fitness = calculateFitness(Population)
        bestFitness[i] = np.max(fitness)# Add the best value in a list
        fitnessProbabilities = adjustedProbabilities(fitness)# Return the probabilities of the fitness

        # sort the fitness and return the indices of the sorting to rearrange the population and take the best two to add them in the new generation
        sortedIndices = np.argsort(fitnessProbabilities)
        sortedPopulation = Population[sortedIndices][::-1]# Rearrange the population in new list sortedPopulation decreasing order

        # Add first element and second element to new generation without change
        newGenerations[0] = sortedPopulation[0]
        newGenerations[1] = sortedPopulation[1]

        # Start adding new children from index two to the end
        j = 2
        while j < populationSize-1:
            parent1, parent2 = selection(Population , fitness)
            child1 , child2 = crossover(parent1, parent2, pCross)

            newGenerations[j] = child1
            newGenerations[j+1] = child2

            j += 2

        # Perform mutation on new generation
        mutation(newGenerations, pMutation)
        Population = newGenerations

    # Return the last generation and best Fitness
    return Population, bestFitness




generationNumber = 100
chromosomeLength = 10# Chromosome length is 10 bit, and each variable takes only 5 bits
mutationRate = 0.05
crossoverRate = 0.6
Pop = np.random.randint(0,2 ,size=(20,10))# Initial population with random values


finalGeneration,bestFitnessHistory = geneticAlgorithm(Pop,generationNumber,chromosomeLength,crossoverRate,mutationRate)


# Print Final generation that came from genetic algorithm and best fitness in each generation
print(finalGeneration)
print(bestFitnessHistory)

# Plot the best fitness in each generation. And save as pdf file
plt.figure(figsize=(8,8))
plt.plot(bestFitnessHistory)
plt.xlabel('Generations')
plt.ylabel('Best Fitness')
plt.title('Best Fitness Over Generations')
plt.savefig('BestFitnessOverGenerations.pdf')

