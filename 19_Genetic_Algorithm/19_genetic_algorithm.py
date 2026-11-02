# Experiment 19: Basic Genetic Algorithm
import random

random.seed(42)

# Maximize f(x) = x^2 for integer x in [0, 31].
POP_SIZE, CHROMOSOME_BITS, GENERATIONS = 20, 5, 40

def fitness(chromosome):
    x = int(chromosome, 2)
    return x * x

def select(population):
    contestants = random.sample(population, 3)
    return max(contestants, key=fitness)

def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_BITS - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(chromosome, rate=0.02):
    bits = list(chromosome)
    for i in range(len(bits)):
        if random.random() < rate:
            bits[i] = "1" if bits[i] == "0" else "0"
    return "".join(bits)

population = ["".join(random.choice("01") for _ in range(CHROMOSOME_BITS))
              for _ in range(POP_SIZE)]
for generation in range(GENERATIONS):
    new_population = [max(population, key=fitness)]  # Elitism
    while len(new_population) < POP_SIZE:
        p1, p2 = select(population), select(population)
        c1, c2 = crossover(p1, p2)
        new_population.extend([mutate(c1), mutate(c2)])
    population = new_population[:POP_SIZE]
    best = max(population, key=fitness)
    print(f"Generation {generation + 1}: x={int(best, 2)}, fitness={fitness(best)}")

best = max(population, key=fitness)
print("Best solution:", int(best, 2), "Fitness:", fitness(best))
