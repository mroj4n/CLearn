#include <iostream>
#include "Knn.hpp"
#include "Dataset.hpp"
#include "AccuracyScore.hpp"


int main() {
    AccuracyScore accuracyScore;
    Dataset dataset(1);
    dataset.load("../DataExamples/classification_data.csv");


    Knn knn(3);
    knn.fit(dataset.getFeatures(), dataset.getLabels());
    std::vector<std::vector<double>> predictions = knn.predict(dataset.getFeatures());
    std::cout << "Accuracy: " << accuracyScore.calculate(predictions, dataset.getLabels()) << std::endl;

    std::cout << "Still works"<< std::endl;
}