"""
@author: Aniruddha Kalburgi
Date: 18 Aug, 2019
"""
from enum import Enum
# from copy import deepcopy
from time import time

import numpy as np
import pandas as pd
import numba as nb
# from pandarallel import pandarallel


class InitializationMethod(Enum):
    RANDOM_INITIALIZATION = 1,


class Chromosome:
    """
    A class to represent Chromosome from the population with its population size.
    Each Chromosome represents a SAT solution, represented by self.values
    The fitness function defines the fitness of the chromosome/potential solution.
    """

    def __init__(self, chromosome_size, genes=None):
        self.chromosome_size = chromosome_size
        self.genes = genes

        if self.genes is None:
            self.initialize_chromozome()

    def initialize_chromozome(self, method: InitializationMethod = InitializationMethod.RANDOM_INITIALIZATION):
        if method is InitializationMethod.RANDOM_INITIALIZATION:
            self.genes = np.random.randint(0, 2, self.chromosome_size)


class CrossoverType(Enum):
    ONE_POINT_CROSSOVER = 1,
    TWO_POINT_CROSSOVER = 2


class MatingStrategy(Enum):
    MUTATION = 'mutation',
    CROSSOVER = 'crossover'


class WeightedMaxSAT_GeneticAlgorithm:
    """
    A class that offers required functionality for the GeneticAlgorithm
    """

    # TODO consider weights of clauses and update the fitnesses accordingly

    def __init__(self, population_size, clauses, clause_weights, chromosome_size,
                 crossover_type=CrossoverType.TWO_POINT_CROSSOVER):
        # the number of variables
        self.chromosome_size = chromosome_size

        # SAT clauses
        self.clauses = clauses
        self.clause_weights = clause_weights

        self.population_size = population_size
        self.population = []
        self.population_fitness = np.zeros(population_size)
        self.crossover_type = crossover_type
        self.best_cost = None
        self.best_costs = []
        self.best_solutions = []

    def calculate_fitness(self, chromosome: Chromosome):
        if chromosome.genes is None:
            chromosome.initialize_chromozome()

        return SATSolver.get_solution_details(chromosome=chromosome, clauses_list=self.clauses,
                                              clause_weights=self.clause_weights)
        # return SATSolver.validate_cnf_formula(chromosome=chromosome, clauses=self.clauses)
        # return SATSolver.parallel_validate_solution(chromosome=chromosome, clauses=self.clauses)

    # def calculate_np_fitness(self):

    def get_fitness_superfast(self):
        return SATSolver.one_shot_population_fitness(population=self.population, clause_weights=self.clause_weights)

    def run(self, max_iter):
        """
        Run the Genetic Algorithm
        :return:
        """
        # initialize the initial population
        # self.initialize_population()
        self.fast_initialize_population()

        # TODO repeat until found the best cost or certain number of iterations
        for i in range(max_iter):
            # old way ------------------------
            # new_best_fitness_index = np.argmin(self.population_fitness)
            new_best_fitness_index = np.argmax(self.population_fitness)
            new_best_fitness = self.population_fitness[new_best_fitness_index]
            #
            if self.best_cost is None or new_best_fitness >= self.best_cost:
                print("Current Best Fitness: ", new_best_fitness)
                self.best_solutions.append(self.population[new_best_fitness_index])
                self.best_cost = new_best_fitness
                self.best_costs.append(self.best_cost)
            #
            self.generate_new_mating_pool()
            # old way ------------------------

            # self.fast_generate_new_mating_pool()

        # print solutions and costs
        print("\nBest Solutions: ")
        for solution in self.best_solutions:
            # print(str(solution.genes.tolist()))
            print(str(solution.tolist()))

        print("\nCosts: ", str(self.best_costs))
        print("Best Cost: ", str(self.best_cost))

    def initialize_population(self):
        for index in range(self.population_size):
            # values are kept None so that Chromosome class does the random initialization
            chromosome = Chromosome(chromosome_size=self.chromosome_size, genes=None)

            # use unsatisfied_clause_indices for weights multiplication with fitness
            solution_cost = self.calculate_fitness(chromosome=chromosome)

            self.population.append(chromosome)
            self.population_fitness[index] = solution_cost

    def fast_initialize_population(self):
        self.population = np.random.randint(0, 2, (self.population_size, self.chromosome_size))

        # SATSolver.column_wise_validate_solution(population=self.population, clauses_df=self.clauses,
        #                                         clause_weights=self.clause_weights,
        #                                         current_fitness_array=self.population_fitness)
        # SATSolver.one_shot_whole_df_truths(population=self.population, clauses_df=self.clauses,
        #                                    clause_weights=self.clause_weights,
        #                                    current_fitness_array=self.population_fitness)

        # self.population_fitness = SATSolver.one_shot_population_fitness(population=self.population,
        #                                                                 clauses_df=self.clauses,
        #                                                                 clause_weights=self.clause_weights,
        #                                                                 current_fitness_array=self.population_fitness)
        self.population_fitness = SATSolver.vectorized_one_shot_population_fitness(population=self.population,
                                                                        clauses_df=self.clauses,
                                                                        clause_weights=self.clause_weights,
                                                                        current_fitness_array=self.population_fitness)

        # self.population_fitness = np.array(self.population_fitness)

    def roulette_wheel_selection(self):
        # calculate selection probability of each chromosome based on their fitness
        fitness_probabilities = self.population_fitness / np.sum(self.population_fitness)

        # select two indices based on fitness probabilities
        index0, index1 = np.random.choice(len(self.population), size=2, p=fitness_probabilities).tolist()

        # select parents at chosen indices
        parent1 = self.population[index0]
        parent2 = self.population[index1]

        return [parent1, parent2]

    def one_point_crossover(self, two_parents):
        parent1 = two_parents[0]
        parent2 = two_parents[1]

        crossover_index = np.random.choice(self.chromosome_size)
        # parent1_genes_after_index = parent1.genes[crossover_index:]
        # parent2_genes_after_index = parent2.genes[crossover_index:]
        #
        # parent1.genes[crossover_index:] = parent2_genes_after_index
        # parent2.genes[crossover_index:] = parent1_genes_after_index
        parent1_genes_after_index = parent1[crossover_index:]
        parent2_genes_after_index = parent2[crossover_index:]

        parent1[crossover_index:] = parent2_genes_after_index
        parent2[crossover_index:] = parent1_genes_after_index

        random_child_choice = np.random.choice(2)

        return [parent1, parent2][random_child_choice]

    def two_point_crossover(self, two_parents):
        child1 = two_parents[0]
        child2 = two_parents[1]

        index1, index2 = tuple(np.sort((np.random.choice(self.chromosome_size, 2))))

        # parent1_genes = child1.genes[index1: (index2 + 1)]
        # parent2_genes = child2.genes[index1: (index2 + 1)]
        #
        # child1.genes[index1: (index2 + 1)] = parent2_genes
        # child2.genes[index1: (index2 + 1)] = parent1_genes

        parent1_genes = child1[index1: (index2 + 1)]
        parent2_genes = child2[index1: (index2 + 1)]

        child1[index1: (index2 + 1)] = parent2_genes
        child2[index1: (index2 + 1)] = parent1_genes

        random_child_choice = np.random.choice(2)

        return [child1, child2][random_child_choice]

    def mutation(self, child):
        # generate probabilities of shape as same as current parent
        probabilities = np.random.sample(self.chromosome_size)

        # mutate/alter genes of a child with 1/L probability
        # child.genes = 1 * np.invert(child.genes == 1, where=probabilities <= 1 / self.chromosome_size)
        child = 1 * np.invert(child == 1, where=probabilities <= 1 / self.chromosome_size)

        return child

    def generate_new_mating_pool(self):
        mating_pool = []
        mating_pool_fitness = []

        # choose two parents for mating using roulette wheel
        random_parents = self.roulette_wheel_selection()

        while len(mating_pool) < self.population_size:
            child = None

            # produce a child using crossover
            if self.crossover_type is CrossoverType.ONE_POINT_CROSSOVER:
                child = self.one_point_crossover(random_parents)
            elif self.crossover_type is CrossoverType.TWO_POINT_CROSSOVER:
                child = self.two_point_crossover(random_parents)

            # mutate/alter a child using mutation
            child = self.mutation(child=child)

            # add the child to the population
            mating_pool.append(child)

        # replace current population with mating pool
        self.population = np.array(mating_pool)

        # update the fitness of child
        # fitness, unsatisfied_clause_indices = self.calculate_fitness(child)
        # fitness = self.calculate_fitness(child)
        # self.get_fitness_superfast()
        # SATSolver.column_wise_validate_solution(population=self.population, clauses_df=self.clauses,
        #                                         clause_weights=self.clause_weights,
        #                                         current_fitness_array=self.population_fitness)
        # SATSolver.one_shot_whole_df_truths(population=self.population, clauses_df=self.clauses,
        #                                    clause_weights=self.clause_weights,
        #                                    current_fitness_array=self.population_fitness)
        # self.population_fitness = SATSolver.one_shot_population_fitness(population=self.population,
        #                                                                 clauses_df=self.clauses,
        #                                                                 clause_weights=self.clause_weights,
        #                                                                 current_fitness_array=self.population_fitness)
        self.population_fitness = SATSolver.vectorized_one_shot_population_fitness(population=self.population,
                                                                                   clauses_df=self.clauses,
                                                                                   clause_weights=self.clause_weights,
                                                                                   current_fitness_array=self.population_fitness)

    def fast_generate_new_mating_pool(self):
        mating_pool = []
        mating_pool_fitness = []

        # choose two parents for mating using roulette wheel
        random_parents = self.roulette_wheel_selection()

        while len(mating_pool) < self.population_size:
            child = None

            # produce a child using crossover
            if self.crossover_type is CrossoverType.ONE_POINT_CROSSOVER:
                child = self.one_point_crossover(random_parents)
            elif self.crossover_type is CrossoverType.TWO_POINT_CROSSOVER:
                child = self.two_point_crossover(random_parents)

            # mutate/alter a child using mutation
            child = self.mutation(child=child)

            # update the fitness of child
            # fitness, unsatisfied_clause_indices = self.calculate_fitness(child)
            # mating_pool_fitness.append(fitness)

            # add the child to the population
            mating_pool.append(child)

        costs, self.population_fitness = self.get_fitness_superfast()

        # replace current population with mating pool
        self.population = mating_pool
        # self.population_fitness = np.array(mating_pool_fitness)


class SATSolver:

    def __init__(self):
        pass

    @staticmethod
    def get_solution_details(chromosome: Chromosome, clauses_list, clause_weights):
        """
        :param chromosome:
        :param clauses_list:
        :return: cost, unsatisfied clause indices
        """

        # unsatisfied_clause_indices = []
        clause_satisfiability_list = []

        for clause in clauses_list:
            # clause = np.array(clause)

            current_solution_values = chromosome.genes[np.absolute(clause) - 1]
            negative_flips = clause < 0
            positive_truths = current_solution_values.astype(np.bool)
            positive_truths[negative_flips] = np.invert(positive_truths[negative_flips])

            clause_satisfiability = any(positive_truths)

            clause_satisfiability_list.append(clause_satisfiability)

            # not clause_satisfiability and unsatisfied_clause_indices.append(index)

        clause_satisfiability_list = np.array(clause_satisfiability_list, dtype=np.int8)
        fitness = np.sum(clause_satisfiability_list * clause_weights)

        return fitness

    @staticmethod
    def column_wise_validate_solution(population, clauses_df, clause_weights, current_fitness_array):
        def find_truths(clause):
            current_solution_values = genes[np.absolute(clause) - 1]
            negative_flips = clause < 0
            positive_truths = current_solution_values.astype(np.bool)
            positive_truths[negative_flips] = np.invert(positive_truths[negative_flips])

            return positive_truths

        for index, genes in enumerate(population):
            # find and store all locations of nan's
            nan_locations = clauses_df.isna()

            # replace all nan's with 1
            clauses_df = clauses_df.fillna(1).astype(np.int)

            # apply truth value fetch on entire df - column wise
            truths_df = clauses_df.apply(lambda x: find_truths(x))

            # replace all known nan location's with False
            truths_df[nan_locations] = False

            # now find truth values of all clauses - see if there's a way to find truth value by axis=1 in Pandas directly
            boolean_series = truths_df.any(axis=1)

            fitness = np.sum(boolean_series * clause_weights)

            current_fitness_array[index] = fitness

    @staticmethod
    def one_shot_whole_df_truths(population, clauses_df, clause_weights, current_fitness_array):
        def one_shot_truths_df(clauses, solution):
            current_solution_values = solution[(np.abs(clauses) - 1)]
            negative_flips = clauses < 0
            positive_truths = current_solution_values.astype(np.bool)
            positive_truths[negative_flips] = np.invert(positive_truths[negative_flips])

            return positive_truths

        for index, genes in enumerate(population):
            # find and store all locations of nan's
            nan_locations = clauses_df.isna()

            # replace all nan's with 1
            clauses_df = clauses_df.fillna(1).astype(np.int)

            # apply truth value fetch on entire df - column wise
            # truths_df = clauses_df.apply(lambda x: find_truths(x))
            truths = one_shot_truths_df(clauses_df.values, genes)
            # truths = np.vectorize(one_shot_truths_df(clauses_df.values, genes))

            # replace all known nan location's with False
            # truths_df[nan_locations] = False
            truths[nan_locations] = False

            # now find truth values of all clauses - see if there's a way to find truth value by axis=1 in Pandas directly
            # boolean_series = truths_df.any(axis=1)
            boolean_np_array = np.any(truths, axis=1)

            # fitness = np.sum(boolean_series * clause_weights)
            fitness = np.sum(boolean_np_array * clause_weights)

            current_fitness_array[index] = fitness

    @staticmethod
    def one_shot_population_fitness(population, clauses_df, clause_weights, current_fitness_array):
        """

        :param population:
        :return:
        """
        # TODO process all population's satisfiability in single shot - either as Pandas Dataframe of Numpy array
        #  and return the costs, population fitness and (indices of unsatisfied clauses - maybe return
        #  the weighted fitness from here only)

        solution = population
        clauses = clauses_df.values

        # find and store all locations of nan's
        nan_locations = np.isnan(clauses)

        # replace all nan's with 1
        clauses[nan_locations] = 1
        clauses = clauses.astype(np.int)

        abs_clauses = np.abs(clauses) - 1
        current_truths = (solution.T[abs_clauses])
        negative_flips = clauses < 0
        p_truths = current_truths.astype(np.bool)
        p_truths[negative_flips] = np.invert(p_truths[negative_flips])

        # replace all known nan location's with False
        # truths_df[nan_locations] = False
        (p_truths)[nan_locations] = False

        clause_satisfaction = np.any(p_truths, axis=1)

        current_fitness_array = np.sum((clause_satisfaction.T * clause_weights), axis=1)

        return current_fitness_array

    @staticmethod
    def vectorized_one_shot_population_fitness(population, clauses_df, clause_weights, current_fitness_array):
        """

        :param population:
        :return:
        """
        # TODO process all population's satisfiability in single shot - either as Pandas Dataframe of Numpy array
        #  and return the costs, population fitness and (indices of unsatisfied clauses - maybe return
        #  the weighted fitness from here only)
        solution = population
        clauses = clauses_df.values

        @nb.vectorize(['bool'], target='cuda')
        def vectorize():
            abs_clauses = np.abs(clauses) - 1
            current_truths = (solution.T[abs_clauses])
            negative_flips = clauses < 0
            p_truths = current_truths.astype(np.bool)
            p_truths[negative_flips] = np.invert(p_truths[negative_flips])

            return p_truths

        # find and store all locations of nan's
        nan_locations = np.isnan(clauses)

        # replace all nan's with 1
        clauses[nan_locations] = 1
        clauses = clauses.astype(np.int)

        # vectorized = np.vectorize(vectorize)
        # p_truths = vectorized()
        p_truths = vectorize()

        # replace all known nan location's with False
        # truths_df[nan_locations] = False
        (p_truths)[nan_locations] = False

        clause_satisfaction = np.any(p_truths, axis=1)

        current_fitness_array = np.sum((clause_satisfaction.T * clause_weights), axis=1)

        return current_fitness_array

    @staticmethod
    def parallel_validate_solution(chromosome: Chromosome, clauses):
        terms = chromosome.genes
        clause_satisfiability_list = []

        clauses = pd.DataFrame.from_records(clauses)

        def apply_per_column(y):
            return not np.isnan(y) and int(y) == terms[np.abs(int(y)) - 1]

        def process_rows(x):
            return x.apply(lambda y: process_clause(y))

        def process_clause(y):
            clause = y[~ np.isnan(y)]
            current_solution_values = chromosome.genes[np.absolute(clause) - 1]
            negative_flips = clause < 0
            positive_truths = current_solution_values.astype(np.bool)
            positive_truths[negative_flips] = np.invert(positive_truths[negative_flips])

            clause_satisfiability = any(positive_truths)

            return clause_satisfiability

        def apply_per_row(x):
            # return x.apply(lambda y: apply_per_column(y))
            return x.apply(lambda y: apply_per_column(y))

        # boolean_df = clauses[:].apply(lambda x: x.apply(lambda y: not np.isnan(y) and int(y) == terms[np.abs(int(y)) - 1]),
        #                       axis=1)
        # boolean_df = clauses.parallel_apply(lambda x: process_rows(x), axis=1)
        boolean_df = clauses.parallel_apply(lambda x: apply_per_row(x), axis=1)

        boolean_series = boolean_df.parallel_apply(lambda x: any(x), axis=1)

        # here we're extracting index of clauses that are not(-) satisfied, that's why the '-' sign
        unsatisfied_clause_indices = clauses[-boolean_series].index.tolist()

        # the length of 'unsatisfied clause index' list is our cost
        cost = len(unsatisfied_clause_indices)

        # return (cost, index of unsatisfied clauses)
        return cost, unsatisfied_clause_indices

    @staticmethod
    def validate_cnf_formula(chromosome: Chromosome, clauses):
        """

        :param terms: this is a solution of 20 variables, indexed from 0 to 19
        :param clauses: list of all clauses
        :return: A tuple representing (all clauses satisfaction condition - boolean,
                                        the cost - int,
                                        indices of unsatisfied clauses - list of clause index)
        """

        terms = chromosome.genes

        # prepare Pandas DataFrame of clauses
        # number of rows will be equal to number of clauses
        # each column contains one variable, number of columns is equal to the number of variables in the lengthiest clause
        df = pd.DataFrame.from_records(clauses)

        # NaN's  are filled in case the variable length of clauses are present
        # outer lambda selects each row i.e. clause, each clause may contain combination of variables and NaN's
        # inner lambda selects each variable in the clause/row and performs following operation
        #   - don't process when identified a NaN
        #   - otherwise, return True if int(variable) has same value in solution/terms array.
        #     -   False if int(variable) != value in solution/terms array
        # all clauses - entire DataFrame is processed by this single line
        boolean_df = df.apply(lambda x: x.apply(lambda y: not np.isnan(y) and int(y) == terms[np.abs(int(y)) - 1]),
                              axis=1)

        # apply boolean OR condition to all clauses/rows
        # the lambda x selects one row/clause at a time
        # in return we get 1D boolean array telling us which row/clauses satisfy and which doesn't
        boolean_series = boolean_df.apply(lambda x: any(x), axis=1)

        # here we're extracting index of clauses that are not(-) satisfied, that's why the '-' sign
        unsatisfied_clause_indices = df[-boolean_series].index.tolist()

        # the length of 'unsatisfied clause index' list is our cost
        cost = len(unsatisfied_clause_indices)

        # return (cost, index of unsatisfied clauses)
        return cost, unsatisfied_clause_indices


def main():
    lines = wcnf_file.readlines()
    config_line = list(filter(lambda x: x.startswith('p'), lines))
    config_line = config_line[0].split()

    num_variables, num_clauses, top = list(map(int, config_line[2:]))

    data_lines = list(filter(lambda x: x.lstrip()[0] not in 'pc', lines))
    string_data_lines = list(map(lambda x: x.split(), data_lines))
    integer_data_lines = list(map(lambda x: list(map(int, x)), string_data_lines))

    previous_lines = []

    clause = []
    clauses_list = []

    for line in integer_data_lines:
        if 0 in line:
            zero_index = line.index(0)

            for prev_line in previous_lines:
                clause.extend(prev_line)

            clause.extend(line[:(zero_index + 1)])

            clauses_list.append(clause)
            clause = []
        else:
            clause.extend(line)

    cleaned_clauses = []
    clause_weights = []

    for clause in clauses_list:
        if len(clause) >= 3:
            weight = clause[0]

            variables = np.array(clause[1:-1])

            cleaned_clauses.append(variables)
            clause_weights.append(weight)

    clauses_df = pd.DataFrame(cleaned_clauses)
    clause_weights = np.array(clause_weights)

    genetic_algorithm = WeightedMaxSAT_GeneticAlgorithm(population_size=population_size, clauses=clauses_df,
                                                        clause_weights=clause_weights, chromosome_size=num_variables,
                                                        crossover_type=crossover_type)
    genetic_algorithm.run(max_iter=max_iterations)


if __name__ == '__main__':
    max_iterations = 10
    # pandarallel.initialize()

    population_size = 200
    crossover_type = CrossoverType.TWO_POINT_CROSSOVER
    # crossover_type = CrossoverType.ONE_POINT_CROSSOVER

    start_time = time()

    files = ['aes-key-recovery-AES1-76-36.wcnf', './aes-mul_8_9.wcnf']
    file = files[0]
    # wcnf_file = open()
    # for index, file in enumerate(files):
    #     # if index == 1:
    #     print("Running for the file: ", file)
    wcnf_file = open(file)
    main()

    finish_time = time() - start_time
    print("Total time taken %.2f seconds" % finish_time)
