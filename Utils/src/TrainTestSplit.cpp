#include "TrainTestSplit.hpp"
#include <vector>


TrainTestSplit::TrainTestSplit(double testSize, const Dataset& dataset)
{
    this->testSize = testSize;
    split(dataset);
}

TrainTestSplit::~TrainTestSplit() {
}

void TrainTestSplit::split(const Dataset& dataset) {
    std::vector<std::vector<double>> features = dataset.getFeatures();
    std::vector<std::vector<double>> labels = dataset.getLabels();
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

Dataset TrainTestSplit::getTrainDataset() const {
    return trainDataset.value();
}

Dataset TrainTestSplit::getTestDataset() const {
    return testDataset.value();
}