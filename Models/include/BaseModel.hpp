#pragma once
#include <vector>
#include <map>
#include <cstdint>
#include <optional>

class BaseModel {
public:
    BaseModel() = default;
    virtual ~BaseModel() = default;
    virtual void fit(const std::vector<std::vector<double>>& features, const std::vector<std::vector<double>>& labels) = 0;
    virtual void partialFit(const std::vector<std::vector<double>>& features, const std::vector<std::vector<double>>& labels) = 0;
    virtual std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& queryData) const = 0;
    virtual std::vector<std::map<double, double>> predictProba(const std::vector<std::vector<double>>& queryData) const = 0;
protected:
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> labels;
};