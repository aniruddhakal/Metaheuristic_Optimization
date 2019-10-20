import random
import sys
from copy import deepcopy
from queue import Queue
from time import time

import matplotlib.pylab as plt
import numpy as np

from CNF_Util import CNF_Loader
from CNF_Util import CNF_Validator


class NoveltyUtil:
    """A Utility class for Novelty+ configuration"""

    def __init__(self, wp=0.4, p_novelty=0.3, iterations_limit=100000, time_limit_minutes=1):
        self.wp = wp
        self.p_for_novelty = p_novelty
        self.iterations_limit = iterations_limit
        self.time_limit = time_limit_minutes
        self.recent_flp_var_per_clause = {}


class GSAT_WithTabu_And_NoveltyPlus:
    """same class runs GSAT with Tabu OR Novelty+, configurable through boolean variable novelty_plus_mode"""

    def __init__(self, CNF_terms_count, CNF_Formula_clauses, max_iterations, max_restarts, tabu_size,
                 novelty_plus_mode=False, verbose_mode=True, novelty_properties=NoveltyUtil()):
        self.CNF_terms_count = CNF_terms_count
        self.CNF_Formula_clauses = CNF_Formula_clauses
        self.max_iterations = max_iterations
        self.max_restarts = max_restarts

        # a dictionary that maintains solutions cost history per restart
        # K - restart number
        # V - list of cost history for whatever number of iterations actually executed
        self.solution_cost_history = {}

        # a boolean variable setting to track whether it's a Novelty+ mode or not
        self.novelty_plus_mode = novelty_plus_mode
        self.verbose_mode = verbose_mode

        self.tabu_size = tabu_size
        # Tabu list - queue with max size as specified - 5 for this run, specified in main() method
        self.tabu = Queue(maxsize=self.tabu_size)

        # settings for Novelty+
        self.novelty_properties = novelty_properties
        self.global_minimum_cost = 99999999

        # set max iterations to that configured for novelty in case Novelty+ mode is on.
        if novelty_plus_mode:
            self.max_iterations = self.novelty_properties.iterations_limit

            # this setting ensures No-restart but only single run for Novelty+
            self.max_restarts = 1

    def __reset(self):
        """Reset values at the start of new restart"""
        self.tabu = Queue(maxsize=self.tabu_size)

        # Novelty+ has no restart but this helps set initial values
        if self.novelty_plus_mode:
            self.novelty_properties.most_recently_flipped_variable = None
            self.novelty_properties.recent_flp_var_per_clause = {}

    def local_search(self):
        """the local search algorithm implementation - includes GSAT_Tabu and Novelty+"""

        for run in range(0, self.max_restarts):
            self.__reset()
            # cost history per restart is maintained by these arrays
            self.solution_cost_history[run] = []
            this_run_cost_history = []

            print("\n\n-----------------# Algorithm restart, Run: %d #-----------------" % run)
            # randomly chosen assignment of variables
            potential_best_term = self.get_initial_terms(self.CNF_terms_count)
            best_term_variable = 1

            # check satisfiability of formula and update following variables
            is_satisfied, cost, unsatisfied_clause_indices = CNF_Validator.validate_cnf_formula(potential_best_term,
                                                                                                self.CNF_Formula_clauses)

            # local search iterations start here
            for step in range(0, self.max_iterations):
                this_run_cost_history.append(cost)
                self.global_minimum_cost = cost

                if self.verbose_mode:
                    print("Iteration: %d" % (step + 1) + "/%d" % self.max_iterations)
                    print("Cost History: %s\n" % str(this_run_cost_history))

                # if current solution satisfies the formula, save cost history for these iterations
                # and return the solution
                if is_satisfied:
                    if self.verbose_mode:
                        print("\n\n-------# Found solution %s #-------" % str(
                            potential_best_term) + ", with cost %d" % cost)

                    self.solution_cost_history[run] = this_run_cost_history
                    self.print_solution_info()
                    return potential_best_term

                # otherwise if Novelty flag is False, execute GSAT with Tabu search
                if not self.novelty_plus_mode:
                    # get next best variable which is not in the Tabu list
                    # update values of best variable, cost, new solution/term and (term_satisfaction) tuple
                    best_term_variable, cost, potential_best_term, term_satisfaction = self.get_next_best_nonTabu_variable(
                        potential_best_term, best_term_variable)

                    if cost == -1:
                        self.solution_cost_history[run] = this_run_cost_history
                        break

                    # extracting values from the (term_satisfaction) tuple
                    # (satisfiability condition as per new best variable,
                    # indices of unsatisfied clauses - not used in GSAT but useful for Novelty+ mode)
                    is_satisfied, unsatisfied_clause_indices = term_satisfaction
                else:
                    # Novelty+
                    is_satisfied, cost, unsatisfied_clause_indices, potential_best_term, best_term_variable = \
                        self.novelty_plus_search(potential_best_term, unsatisfied_clause_indices)

            self.solution_cost_history[run] = this_run_cost_history
            print("Run %d solutions cost" % run + " %s" % str(this_run_cost_history))

        self.print_solution_info()
        print("Max Runs %d reached... Stopping the algorithm!" % self.max_restarts)

    def novelty_plus_search(self, potential_best_term, unsatisfied_clause_indices):
        best_term_variable = None

        # 1. Select an unsatisfied clause c (uniformly at random)
        rndm_unsat_cl_idx = np.random.choice(unsatisfied_clause_indices)
        random_unsatisfied_clause = self.CNF_Formula_clauses[rndm_unsat_cl_idx]

        # 2. With some probability wp select a random variable from c,
        # while in the remaining cases use Novelty for your variable selection process.
        if random.random() <= self.novelty_properties.wp:
            # select random variable from c
            random_variable = np.random.choice(random_unsatisfied_clause)
            if self.verbose_mode:
                print("Flipping random variable %d" % random_variable + " from unsatisfied clause %s" % str(
                    random_unsatisfied_clause))

            # add abs(variable) to the most recently flipped record of currently selected clause
            self.novelty_properties.recent_flp_var_per_clause[
                rndm_unsat_cl_idx] = np.abs(random_variable)
            # self.novelty_properties.recently_flipped_var = np.abs(random_variable)

            # flip the variable value
            potential_best_term[np.abs(random_variable) - 1] *= -1

            # update the cost and other values after new variable flip
            is_satisfied, cost, unsatisfied_clause_indices = CNF_Validator.validate_cnf_formula(
                potential_best_term,
                self.CNF_Formula_clauses)
        else:
            # Novelty:​​ Identify the best V​ best​ and second best variables V​ 2best​ in c (w.r.t.
            # the minimization of the number of unsatisfied clauses).
            novelty_selected_variable = None

            # identify v-best and v-second-best
            v_best, v_second_best = self.get_v_best_and_second_best(random_unsatisfied_clause,
                                                                    potential_best_term)

            # If V​ best​ is not the most recently flipped variable in the clause, novelty will select this variable.
            if (rndm_unsat_cl_idx not in self.novelty_properties.recent_flp_var_per_clause.keys()) or \
                    (self.novelty_properties.recent_flp_var_per_clause[rndm_unsat_cl_idx] != v_best[0]):
                novelty_selected_variable = v_best[0]
                is_satisfied, cost, unsatisfied_clause_indices = v_best[2]
            else:  # Otherwise, V​ 2best will
                if random.random() <= 0.3:  # be flipped with some probability p i.e. 30%
                    # considering v_best and v_second_best are the tuples with these values:
                    #           (v_best, v_best_term, v_best_validation_result),
                    #           (v_second_best, v_second_best_term, v_second_best_validation_result)
                    #
                    # therefore [0]th element is actual best_variable, [1]st element is the entire solution and
                    # [2]nd element is validation_result tuple...!
                    novelty_selected_variable = v_second_best[0]

                    # extract pre-calculated cost and other values after new variable flip
                    is_satisfied, cost, unsatisfied_clause_indices = v_second_best[2]
                else:  # and V​ best with some probability 1-p i.e. 70%.
                    # select v1 with 70% chance
                    novelty_selected_variable = v_best[0]

                    # extract pre-calculated cost and other values after new variable flip
                    is_satisfied, cost, unsatisfied_clause_indices = v_best[2]

            # add the abs(variable) to the recently flipped clause under index of this clause as a key
            # this is to maintain the track of most recently flipped variable per clause...!
            self.novelty_properties.recent_flp_var_per_clause[rndm_unsat_cl_idx] = np.abs(novelty_selected_variable)

            # flip the variable in potential best term, relative values of potential_best_term etc are already updated
            best_term_variable = novelty_selected_variable
            potential_best_term[np.abs(novelty_selected_variable) - 1] *= -1

        return is_satisfied, cost, unsatisfied_clause_indices, potential_best_term, best_term_variable

    def get_v_best_and_second_best(self, unsatisfied_clause, seed_term):
        costs = []
        terms = []
        validation_results = []

        for variable in unsatisfied_clause:
            # flip the i-th variable of solution
            seed_term[(np.abs(variable) - 1)] *= -1

            # validate new solution and get tuple (satisfiability, cost and unsatisfied clauses indices)
            validation_result = CNF_Validator.validate_cnf_formula(seed_term, self.CNF_Formula_clauses)
            # un-flip the i-th variable of solution
            seed_term[(np.abs(variable) - 1)] *= -1

            # extract necessary information from tuple
            cost = validation_result[1]
            costs.append(cost)
            terms.append(seed_term)
            validation_results.append(validation_result)

        sorted_indices = np.argsort(np.array(costs))

        v_best = unsatisfied_clause[sorted_indices[0]]
        v_best_term = terms[sorted_indices[0]]
        v_best_validation_result = validation_results[sorted_indices[0]]

        # case - when there's only 1 variable in unsatisfied clause, choosing v_best as second_best for now
        # select and put best_terms for v_best and v_second_best so that they don't have to be recomputed
        if len(unsatisfied_clause) > 1:
            v_second_best = unsatisfied_clause[sorted_indices[1]]
            v_second_best_term = terms[sorted_indices[1]]
            v_second_best_validation_result = validation_results[sorted_indices[1]]
        else:
            v_second_best = v_best
            v_second_best_term = v_best_term
            v_second_best_validation_result = v_best_validation_result

        return (v_best, v_best_term, v_best_validation_result), (
            v_second_best, v_second_best_term, v_second_best_validation_result)

    def print_solution_info(self):
        print("All Runs - Cost History: %s" % str(self.solution_cost_history))

    def get_next_best_nonTabu_variable(self, current_best_solution, best_term_variable):
        # because inside this function, solution index are handled from 0-19, but 1-20 outside
        best_term_variable -= 1

        costs_dict = {}
        new_best_cost = self.global_minimum_cost

        for i in range(0, self.CNF_terms_count):
            # consider variable in potential best list only if it's not in the Tabu list
            if (i + 1) not in self.tabu.queue:  # this will only calculate costs for variables not-in tabu-list
                # flip variable i
                current_best_solution[i] *= -1

                # get new cost
                the_tuple = CNF_Validator.validate_cnf_formula(current_best_solution, self.CNF_Formula_clauses)
                new_cost = the_tuple[1]

                # this will only add cost if it's less or equals to global_minimum - saves additional checks...
                if new_cost <= self.global_minimum_cost and new_cost <= new_best_cost:
                    new_best_cost = new_cost

                    # get current list
                    this_cost_vars = []

                    # get existing list of costs for 'new_cost'
                    if new_best_cost in costs_dict:
                        this_cost_vars = costs_dict[new_best_cost]

                    this_cost_vars.append(i)  # append variable to the list of same costs

                    # maintain dictionary of <K,V> where K - Cost & V - list of variables which yields same cost
                    costs_dict[new_best_cost] = this_cost_vars

                # un-flip the variable i to restore current_best_solution
                current_best_solution[i] *= -1

        # select best cost variable
        sorted_costs = sorted(costs_dict.keys())

        new_solution = current_best_solution

        # this means improvement is possible
        if len(sorted_costs) != 0:
            new_cost_var_list = costs_dict[sorted_costs[0]]

            # in case of tie for best cost - randomly choose the new best variable for best possible cost
            best_var = np.random.choice(new_cost_var_list)

            # flip best variable in new solution
            new_solution[best_var] *= -1

            # add variable to the tabu_list
            if self.tabu.full():
                self.tabu.get()

            self.tabu.put((best_var + 1))

            print("Tabu List: " + str(self.tabu.queue))
            print("Best Variable Choice: " + str(best_var + 1))

            # get updated cost and other properties for new solution
            is_satisfied, cost, unsatisfied_clause_indices = CNF_Validator.validate_cnf_formula(new_solution,
                                                                                                self.CNF_Formula_clauses)
            return (best_var + 1), sorted_costs[0], new_solution, (is_satisfied, unsatisfied_clause_indices)
        else:
            # this means no further improvement is possible
            # -1 is handled outside this method to force restart in such situations
            return None, -1, None, None  # force restart

    def get_initial_terms(self, CNF_terms_count):
        terms = []
        for i in range(1, CNF_terms_count + 1):
            if random.random() >= 0.5:
                i *= -1

            terms.append(i)

        return terms


def plot_and_save_solution_graph(cost_history, fig_output_path, figure_name, save_flag):
    plt.clf()
    for key in cost_history:
        plt.plot(cost_history[key])

    keys = ["Restart %d" % k for k in cost_history.keys()]
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend(keys)
    plt.title(figure_name)

    output_path = fig_output_path + "/" + figure_name

    if save_flag:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()


def main():
    # False - Run GSAT with Tabu search
    # True - Run Novelty+
    novelty_plus_mode = False

    # Make False to avoid extra printing
    verbose_mode = True

    max_restarts = 10

    # False - show graph for the run
    # True - save graph for the run at specified location
    save_graph_flag = False

    if len(sys.argv) < 1:
        raise Exception("Please provide at least instance file name as an argument.")

    # clause_file = "./input/inst/uf20-20.cnf"
    # output_path = "./output"
    # output_file_name = "novelty_per_clause"
    clause_file = sys.argv[1]

    # naming output directory and graph file
    if len(sys.argv) > 1:
        output_path = sys.argv[2]
    else:
        output_path = "./"

    if len(sys.argv) > 2:
        output_file_name = sys.argv[3]
    else:
        output_file_name = "output"

    # extracting the list of clauses from the clause_file
    terms_count, clauses_count, clauses = CNF_Loader.prepare_clauses(clause_file)

    # if verbose_mode:
    #     print("Clauses: %s" % str(clauses))

    # ---------------------------GSAT with Tabu Search---------------#
    tabu_size = 5
    max_iterations = 1000
    novelty_plus_mode = False
    verbose_mode = True
    max_restarts = 10

    start = time()

    # run the local search algorithm with settings
    gsat_tabu_novelty_plus = GSAT_WithTabu_And_NoveltyPlus(terms_count, clauses, max_iterations, max_restarts,
                                                           tabu_size,
                                                           novelty_plus_mode,
                                                           verbose_mode)
    gsat_tabu_novelty_plus.local_search()

    finish = time() - start

    # plot
    op_file = output_file_name + "_tabu"
    plot_and_save_solution_graph(gsat_tabu_novelty_plus.solution_cost_history, output_path, op_file, save_graph_flag)

    print("\n\nTotal time Tabu: %f seconds" % finish)

    # ---------------------------Novelty+ Mode---------------#
    novelty_plus_mode = True
    # max iterations are set by constructor in Novelty+

    start = time()

    # run the local search algorithm with settings
    gsat_tabu_novelty_plus = GSAT_WithTabu_And_NoveltyPlus(terms_count, clauses, max_iterations, max_restarts,
                                                           tabu_size,
                                                           novelty_plus_mode,
                                                           verbose_mode)
    gsat_tabu_novelty_plus.local_search()

    finish = time() - start

    # plot
    op_file = output_file_name + "_novelty"
    plot_and_save_solution_graph(gsat_tabu_novelty_plus.solution_cost_history, output_path, op_file, save_graph_flag)

    print("\n\nTotal time for Novelty+: %f seconds" % finish)


if __name__ == '__main__':
    main()
