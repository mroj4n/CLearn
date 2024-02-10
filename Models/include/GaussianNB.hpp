#pragma once
#include "BaseModel.hpp"
#include <limits>
class GaussianNB : public BaseModel {
public:
    GaussianNB(std::optional<std::vector<double>> classPriorProbabilities = std::nullopt, double smoothingFactor = std::numeric_limits<double>::epsilon(), bool featureScaling = false);
    ~GaussianNB();
    void fit(const std::vector<std::vector<double>>& features, const std::vector<std::vector<double>>& labels) override;
    void partialFit(const std::vector<std::vector<double>>& features, const std::vector<std::vector<double>>& labels) override;
    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& queryData) const override;
    std::vector<std::map<double, double>> predictProba(const std::vector<std::vector<double>>& queryData) const override;

    // Serialize function for boost
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar& features;
        ar& labels;
        ar& classPriorProbabilities;
        ar& classMeans;
        ar& classVariances;
        ar& numOfClasses;
    }
private:
    void calculateClassPriorProbabilities();
    void calculateClassMeans();
    void calculateClassVariances();
    double calculateLikelyhoodForFeature(const std::vector<double>& feature, const std::vector<double>& classMean, const std::vector<double>& classVariance) const;
    double calculateJointLogLikelyhood(const std::vector<double>& feature, const std::vector<double>& classMean, const std::vector<double>& classVariance) const;
    std::vector<std::map<double, double>> calculatePosteriorProbabilities(const std::vector<std::vector<double>>& queryData) const;
    void normalizeProbabilities(std::vector<std::map<double, double>>& probabilities) const;
    void scaleFeatures(std::vector<std::vector<double>>& features);
    std::optional<std::vector<double>> classPriorProbabilities;
    std::vector<std::map<double, double>> logProbabilities;
    std::vector<std::vector<double>> classMeans;
    std::vector<std::vector<double>> classVariances;
    uint16_t numOfClasses = 0;
    std::vector<uint16_t> numberOfTrainingSamplesInEachClass;
    double smoothingFactor;
    bool featureScaling;
};
