# Genetic algorithms
import numpy as nmp
import matplotlib.pyplot as mtplt
import random

population_size = 10
chromosome_length = 8
generations = 50
mutation_rate = 0.1


def initialize_population(population_size, chromosome_length):
    return [[random.randint(0, 1) for _ in range(chromosome_length)] for _ in range(population_size)]


def fitness(chromosome):
    # Implement your fitness function here
    return sum(chromosome)


def selection(population):
    # Select two parents based on their fitness
    return random.choices(population, weights=[fitness(chromosome) for chromosome in population], k=2)


def crossover(parent1, parent2):
    # Implement crossover (e.g., single-point crossover)
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutation(chromosome, mut_rate):
    # Implement mutation
    mutated_chromosome = [bit ^ (random.random() < mut_rate) for bit in chromosome]
    return mutated_chromosome


def genetic_algorithm(population_size, chromosome_length, generations, mutation_rate):
    population = initialize_population(population_size, chromosome_length)
    for generation in range(generations):
        new_population = []
    for _ in range(population_size // 2):
        parent1, parent2 = selection(population)
    child1, child2 = crossover(parent1, parent2)
    child1 = mutation(child1, mutation_rate)
    child2 = mutation(child2, mutation_rate)
    new_population.extend([child1, child2])
    population = new_population

    # Optional: return the best individual from the final population
    best_individual = max(population, key=fitness)
    return best_individual


# Example usage
best_individual = genetic_algorithm(population_size, chromosome_length, generations, mutation_rate)
print("Best Individual:", best_individual)
print("Fitness:", fitness(best_individual))


# Function for implementing the single-point crossover
def crossover0(l1, q1):
    # Converting the string to list for performing the crossover
    l1 = list(l1)
    q1 = list(q1)

# generating the random number to perform crossover
    k = random.randint(0, 15)
    print("Crossover point :", k)

# interchanging the genes
    for i in range(k, len(s)):
        l1[i], q1[i] = q1[i], l1[i]
        l1 = ''.join(l1)
        q1 = ''.join(q1)
        print(l1)
        print(q1, "\n\n")
        return l1, q1


# patent chromosomes:

s = '1100110110110011'
p = '1000110011011111'
print("Parents")
print("P1 :", s)
print("P2 :", p, "\n")

# function calling and storing the off springs for
# next generation crossover
for i in range(5):
    print("Generation ", i+1, "Childrens :")
    s, p = crossover0(s, p)

# Regression program


def estimate_coeff(p, q):
    # Here, we will estimate the total number of points or observation
    n1 = nmp.size(p)
# Now, we will calculate the mean of a and b vector
    m_p = nmp.mean(p)
    m_q = nmp.mean(q)

# here, we will calculate the cross deviation and deviation about a
    ss_pq = nmp.sum(q * p) - n1 * m_q * m_p
    ss_pp = nmp.sum(p * p) - n1 * m_p * m_p

# here, we will calculate the regression coefficients
    b_1 = ss_pq / ss_pp
    b_0 = m_q - b_1 * m_p

    return b_0, b_1


def plot_regression_line(p, q, b):
    # Now, we will plot the actual points or observation as scatter plot
    mtplt.scatter(p, q, color="m", marker="o", s=30)

# here, we will calculate the predicted response vector
    q_pred = b[0] + b[1] * p

# here, we will plot the regression line
    mtplt.plot(p, q_pred, color="g")

# here, we will put the labels
    mtplt.xlabel('p')
    mtplt.ylabel('q')

# here, we will define the function to show plot
    mtplt.show()


def main():
    # Entering the observation points or data
    p = nmp.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    q = nmp.array([11, 13, 12, 15, 17, 18, 18, 19, 20, 22])

# now, we will estimate the coefficients
    b = estimate_coeff(p, q)
    print("Estimated coefficients are :\nb_0 = {} \nb_1 = {}".format(b[0], b[1]))

# Now, we will plot the regression line
    plot_regression_line(p, q, b)


if __name__ == "__main__":
    main()
