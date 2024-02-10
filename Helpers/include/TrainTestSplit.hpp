#pragma once
#include "Dataset.hpp"
#include <optional>
#include <random>
class TrainTestSplit {
public:
    TrainTestSplit(double testSize, const Dataset& dataset, std::optional<int> randomSeed = std::nullopt);
    ~TrainTestSplit();
    Dataset getTrainDataset() const;
    Dataset getTestDataset() const;
private:
    void split(const Dataset& dataset);
    std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> randomlyShuffleData (std::vector<std::vector<double>> features, std::vector<std::vector<double>> labels);
    double testSize;
    std::optional<Dataset> trainDataset;
    std::optional<Dataset> testDataset;
    int randomSeed;
};