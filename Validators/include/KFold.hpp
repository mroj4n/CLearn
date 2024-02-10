#pragma once
#include "Dataset.hpp"
#include <vector>
#include <optional>
class KFold {
public:
    KFold(const Dataset& dataset, const uint16_t& numOfFolds, std::optional<int> randomSeed = std::nullopt);
    ~KFold();
    std::vector<std::vector<uint16_t>> getIndices() const;
    std::vector<std::pair<Dataset,Dataset>> getDatasetsForEachFold() const;
private:
    std::vector<std::vector<uint16_t>> indicesForEachFold;
    const Dataset& dataset;
    uint16_t numOfFolds;
};
