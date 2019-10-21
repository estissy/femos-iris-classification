from random import seed
from statistics import mean

from femos.core import get_number_of_nn_weights, get_evolved_population
from femos.genotypes import SimpleGenotype
from femos.phenotypes import Phenotype
from femos.selections import get_two_size_tournament_parent_selection, get_age_based_offspring_selection
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data_path = "data/iris.csv"
test_size = 0.3
input_nodes = 4
hidden_layers_nodes = []
output_nodes = 3
number_of_nn_weights = get_number_of_nn_weights(input_nodes, hidden_layers_nodes, output_nodes)
weight_lower_threshold = -1
weight_upper_threshold = 1
population_size = 20
mutation_mean = 0
mutation_standard_deviation = 0.1
epochs = 1000
random_state = 777

# Set up random seed for recurrence of the experiments
seed(random_state)

raw_data = read_csv(data_path)
raw_data_features = raw_data[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
raw_data_output = raw_data["variety"]

data_features = raw_data_features.to_numpy()
data_outputs = raw_data_output.to_numpy()

label_encoder = LabelEncoder()
label_encoder.fit(data_outputs)
transformed_data_outputs = label_encoder.transform(data_outputs)

x_train, x_test, y_train, y_test = train_test_split(data_features, transformed_data_outputs, test_size=test_size,
                                                    shuffle=True, random_state=random_state)


def phenotype_strategy(genotype):
    return Phenotype.get_phenotype_from_genotype(genotype, input_nodes, hidden_layers_nodes, output_nodes)


def evaluation_strategy(phenotypes):
    phenotype_values = []
    test_phenotype_values = []

    for index in range(population_size):
        selected_phenotype = phenotypes[index]
        raw_prediction = Phenotype.get_prediction(selected_phenotype, x_train)
        parsed_prediction = argmax(raw_prediction, axis=1)

        score = 0
        total_rows = len(y_train)

        for row_index in range(total_rows):
            if y_train[row_index] == parsed_prediction[row_index]:
                score += 1

        phenotype_values.append(score / total_rows)

        test_score = 0
        total_test_rows = len(x_test)

        test_raw_prediction = Phenotype.get_prediction(selected_phenotype, x_test)
        test_parsed_prediction = argmax(test_raw_prediction, axis=1)

        for row_index in range(total_test_rows):
            if y_test[row_index] == test_parsed_prediction[row_index]:
                test_score += 1

        test_phenotype_values.append(test_score / total_test_rows)

    print("Correct prediction fraction on test set:", mean(test_phenotype_values))
    return phenotype_values


def parent_selection_strategy(phenotypes_values):
    return get_two_size_tournament_parent_selection(phenotypes_values, population_size)


def mutation_strategy(genotype):
    return SimpleGenotype.get_mutated_genotype(genotype, mutation_mean, mutation_standard_deviation)


def offspring_selection_strategy(parents, updated_parents):
    return get_age_based_offspring_selection(parents, updated_parents)


initial_population = SimpleGenotype.get_random_genotypes(population_size, number_of_nn_weights, weight_lower_threshold,
                                                         weight_upper_threshold)

print("Evolving population of neural network coded as Simple Genotype")
evolved_population = get_evolved_population(initial_population, phenotype_strategy, evaluation_strategy,
                                            parent_selection_strategy, mutation_strategy, offspring_selection_strategy,
                                            epochs)
