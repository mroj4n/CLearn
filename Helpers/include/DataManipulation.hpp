#pragma once
#include "Dataset.hpp"
#include <vector>
#include <random>
#include <optional>
#include <algorithm>
namespace DataManipulation
{
static std::vector<uint16_t> randomizeIndexes (std::vector<uint16_t> indexes, std::optional<int> randomSeed = std::nullopt)
{
    std::random_device rd;
    if (!randomSeed.has_value())
        randomSeed = rd();
    std::mt19937 g(randomSeed.value());
    std::shuffle(indexes.begin(), indexes.end(), g);
    return indexes;
}

} // namespace DataManipulation