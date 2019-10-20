import re
import pandas as pd
import numpy as np
from time import time


class CNF_Validator:
    def __init__(self, terms, clauses):
        pass

    @staticmethod
    def validate_cnf_formula(terms, clauses):
        """

        :param terms: this is a solution of 20 variables, indexed from 0 to 19
        :param clauses: list of all clauses
        :return: A tuple representing (all clauses satisfaction condition - boolean,
                                        the cost - int,
                                        indices of unsatisfied clauses - list of clause index)
        """

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

        #return (satisfiability, cost, index of unsatisfied clauses)
        return cost == 0, cost, unsatisfied_clause_indices


class CNF_Loader:
    def __init__(self):
        pass

    @staticmethod
    def prepare_terms(terms_file):
        string = open(terms_file).read()
        rx = re.compile("v\s.*0")
        term_lines = rx.findall(string)

        r2 = re.compile(r"-?\d{1,}")
        term_strings = []
        for line in term_lines:
            term_strings.extend(r2.findall(line))

        # map string values to integer
        terms = list(map(int, term_strings))

        # return all but last value as it is 0 - an end of terms indicator
        return terms[:-1]

    @staticmethod
    def prepare_clauses(clauses_file):
        string = open(clauses_file).read()
        rx = re.compile("^\s*?[p\d{1,}-].*$", re.MULTILINE)
        term_lines = rx.findall(string, re.MULTILINE)
        line1 = term_lines[0].split()
        terms_count = int(line1[2])
        clauses_count = int(line1[3])
        clauses = []

        for line in term_lines[1:-1]:
            clauses.append(list(map(int, line.split()[:-1])))

        return terms_count, clauses_count, clauses


def main():
    # only to use this validator independently,
    # otherwise can pass a list of terms and clauses to the class and can get solution directly
    terms_file = "./input/sols/9.txt"
    clauses_file = "./input/inst/uf20-09.cnf"

    terms_count, clauses_count, clauses = CNF_Loader.prepare_clauses(clauses_file)
    terms = CNF_Loader.prepare_terms(terms_file)

    print(clauses)
    print(terms)

    if terms_count != len(terms):
        raise Exception(
            "Count of terms doesn't match with that mentioned in clauses file.\nExpected %d" % terms_count + ", found: %d" % len(
                terms))

    start = time()
    is_satisfied, cost, unsatisfied_clauses = CNF_Validator.validate_cnf_formula(terms=terms, clauses=clauses)

    finish = time() - start

    print("Satisfiability: " + str(is_satisfied))
    print("Cost: %d" % cost)
    print("Unsatisfied Clauses: " + str(unsatisfied_clauses))
    print("Time: %f" % finish)


if __name__ == '__main__':
    main()
