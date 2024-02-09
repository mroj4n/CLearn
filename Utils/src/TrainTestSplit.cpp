#include "TrainTestSplit.hpp"
#include <vector>
#include <algorithm>

TrainTestSplit::TrainTestSplit(double testSize, const Dataset& dataset, std::optional<int> randomSeed)
{
    std::random_device rd;
    if (!randomSeed.has_value())
        this->randomSeed = rd();
    else
        this->randomSeed = randomSeed.value();
    this->testSize = testSize;
    split(dataset);
}

TrainTestSplit::~TrainTestSplit() {
}

void TrainTestSplit::split(const Dataset& dataset) {
    std::vector<std::vector<double>> features = dataset.getFeatures();
    std::vector<std::vector<double>> labels = dataset.getLabels();

    std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> shuffledData = randomlyShuffleData(features, labels);
    features = shuffledData.first;
    labels = shuffledData.second;

    std::vector<std::string> labelNames = dataset.getLabelNames();
    std::vector<std::string> featureNames = dataset.getFeatureNames();
    int numLabels = dataset.getNumLabels();
    bool headerExists = dataset.getHeaderExists();
    char delimiter = dataset.getDelimiter();

    std::vector<std::vector<double>> trainFeatures;
    std::vector<std::vector<double>> testFeatures;
    std::vector<std::vector<double>> trainLabels;
    std::vector<std::vector<double>> testLabels;

    int testSizeInt = (int) (features.size() * testSize);
    for (int i = 0; i < features.size(); i++) {
        if (i < testSizeInt) {
            testFeatures.push_back(features[i]);
            testLabels.push_back(labels[i]);
        } else {
            trainFeatures.push_back(features[i]);
            trainLabels.push_back(labels[i]);
        }
    }

    trainDataset.emplace(numLabels, trainFeatures, trainLabels, labelNames, featureNames, headerExists, delimiter);
    testDataset.emplace(numLabels, testFeatures, testLabels, labelNames, featureNames, headerExists, delimiter);

}

std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> TrainTestSplit::randomlyShuffleData (std::vector<std::vector<double>> features, std::vector<std::vector<double>> labels) {
    std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> shuffledData;
    std::vector<int> indices;
    for (int i = 0; i < features.size(); i++) {
        indices.push_back(i);
    }
    std::random_device rd;
    std::mt19937 g(randomSeed);
    std::shuffle(indices.begin(), indices.end(), g);
    for (int i = 0; i < features.size(); i++) {
        shuffledData.first.push_back(features[indices[i]]);
        shuffledData.second.push_back(labels[indices[i]]);
    }
    return shuffledData;
}

Dataset TrainTestSplit::getTrainDataset() const {
    return trainDataset.value();
}

Dataset TrainTestSplit::getTestDataset() const {
    return testDataset.value();
}