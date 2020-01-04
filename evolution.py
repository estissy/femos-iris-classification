from random import seed

from femos.parser import handle_evolution_run
from femos.phenotypes import Phenotype
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data_path = "data/iris.csv"
test_size = 0.3
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


def evaluation_strategy(phenotypes):
    population_size = len(phenotypes)
    phenotype_values = []

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
    return phenotype_values


handle_evolution_run(evaluation_strategy)
