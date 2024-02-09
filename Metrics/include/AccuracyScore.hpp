#pragma once

#include <vector>

class AccuracyScore {
public:
    static double calculate(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& actual);
};