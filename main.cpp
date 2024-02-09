#include <iostream>
#include "Knn.hpp"
#include "Dataset.hpp"
#include "AccuracyScore.hpp"
#include "TrainTestSplit.hpp"

int main() {
    AccuracyScore accuracyScore;
    Dataset dataset(1);
    dataset.load("../DataExamples/classification_data.csv");
    TrainTestSplit trainTestSplit(0.2, dataset);
    Dataset trainDataset = trainTestSplit.getTrainDataset();
    Dataset testDataset = trainTestSplit.getTestDataset();

    Knn knn(3);


    knn.fit(trainDataset.getFeatures(), trainDataset.getLabels());
    std::vector<std::vector<double>> predictions = knn.predict(testDataset.getFeatures());
    std::cout << "Accuracy: " << accuracyScore.calculate(predictions, testDataset.getLabels()) << std::endl;

    std::cout << "Still works"<< std::endl;
}