#pragma once
#include "Knn.hpp"
#include "Dataset.hpp"
#include "AccuracyScore.hpp"
#include "TrainTestSplit.hpp"
#include "ModelSaver.hpp"
#include "KFold.hpp"

namespace testingImplementations {
double doKnnGetAccuracy(Dataset trainDataset, Dataset testDataset)
{
    AccuracyScore accuracyScore;
    Knn knn(3);

    knn.fit(trainDataset.getFeatures(), trainDataset.getLabels());
    std::vector<std::vector<double>> predictions = knn.predict(testDataset.getFeatures());
    return accuracyScore.calculate(predictions, testDataset.getLabels());
}

void testKFold()
{
    Dataset dataset(1);
    dataset.load("../DataExamples/easy_data.csv");

    KFold kFold(dataset, 3);
    std::vector<std::pair<Dataset, Dataset>> datasetsSplitted2 = kFold.getDatasetsForEachFold();

    std::vector<double> accuracies;
    for (int i = 0; i < datasetsSplitted2.size(); i++)
    {
        accuracies.push_back(doKnnGetAccuracy(datasetsSplitted2[i].first, datasetsSplitted2[i].second));
    }

    for (int i = 0; i < accuracies.size(); i++)
    {
        std::cout << "Accuracy for fold:" << i << " is:" << accuracies[i] << std::endl;
    }

}


void testReloading()
{
    AccuracyScore accuracyScore;
    Dataset dataset(1);
    dataset.load("../DataExamples/classification_data.csv");
    TrainTestSplit trainTestSplit(0.2, dataset, 42);
    Dataset trainDataset = trainTestSplit.getTrainDataset();
    Dataset testDataset = trainTestSplit.getTestDataset();

    Knn knn(3);


    knn.fit(trainDataset.getFeatures(), trainDataset.getLabels());
    std::vector<std::vector<double>> predictions = knn.predict(testDataset.getFeatures());
    std::cout << "Accuracy: " << accuracyScore.calculate(predictions, testDataset.getLabels()) << std::endl;

    // Save and reload model and recheck accuracy
    Utils::ModelSaver::saveModel(knn, "knn_model");
    Knn knnReloaded = Utils::ModelSaver::loadModel<Knn>("knn_model");
    std::vector<std::vector<double>> predictionsReloaded = knnReloaded.predict(testDataset.getFeatures());
    std::cout << "Accuracy reloaded: " << accuracyScore.calculate(predictionsReloaded, testDataset.getLabels()) << std::endl;
}
}