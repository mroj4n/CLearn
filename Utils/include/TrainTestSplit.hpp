#pragma once
#include "Dataset.hpp"
#include <optional>

class TrainTestSplit {
public:
    TrainTestSplit(double testSize, const Dataset& dataset);
    ~TrainTestSplit();
    Dataset getTrainDataset() const;
    Dataset getTestDataset() const;
private:
    void split(const Dataset& dataset);
    double testSize;
    std::optional<Dataset> trainDataset;
    std::optional<Dataset> testDataset;
};