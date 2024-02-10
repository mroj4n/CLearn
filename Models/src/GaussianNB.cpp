#include "GaussianNB.hpp"
#include <cmath>
#include <algorithm>

#define M_PI 3.14159265358979323846

GaussianNB::GaussianNB(std::optional<std::vector<double>> classPriorProbabilities, double smoothingFactor, bool featureScaling) : classPriorProbabilities(classPriorProbabilities), smoothingFactor(smoothingFactor), featureScaling(featureScaling) {}
GaussianNB::~GaussianNB() {}

void GaussianNB::partialFit(const std::vector<std::vector<double>>& features, const std::vector<std::vector<double>>& labels) {
    this->features.insert(this->features.end(), features.begin(), features.end());
    this->labels.insert(this->labels.end(), labels.begin(), labels.end());
    if (featureScaling) {
        scaleFeatures(this->features);
    }
    calculateClassPriorProbabilities();
    calculateClassMeans();
    calculateClassVariances();
}

void GaussianNB::fit(const std::vector<std::vector<double>>& features, const std::vector<std::vector<double>>& labels) {
    //clear and call partial fit
    numOfClasses = 0;
    classMeans.clear();
    classVariances.clear();
    logProbabilities.clear();
    numberOfTrainingSamplesInEachClass.clear();
    this->features.clear();
    this->labels.clear();

    partialFit(features, labels);
}

std::vector<std::map<double, double>> GaussianNB::predictProba(const std::vector<std::vector<double>>& queryData) const {
    std::vector<std::map<double, double>> probabilities = calculatePosteriorProbabilities(queryData);
    normalizeProbabilities(probabilities);
    return probabilities;
}

std::vector<std::vector<double>> GaussianNB::predict(const std::vector<std::vector<double>>& queryData) const {
    std::vector<std::map<double, double>> probabilities = predictProba(queryData);
    std::vector<std::vector<double>> predictions;
    for (const auto& probability : probabilities) {
        double maxProbability = std::numeric_limits<double>::min();
        double bestClass = -1;
        for (const auto& [label, prob] : probability) {
            if (prob > maxProbability) {
                maxProbability = prob;
                bestClass = label;
            }
        }
        predictions.push_back({ bestClass });
    }

    return predictions;
}

void GaussianNB::calculateClassPriorProbabilities() {
    if (classPriorProbabilities.has_value()) {
        return;
    }
    std::vector<double> uniqueClasses;
    for (const auto& label : labels) {
        if (std::find(uniqueClasses.begin(), uniqueClasses.end(), label[0]) == uniqueClasses.end()) {
            uniqueClasses.push_back(label[0]);
        }
    }
    numOfClasses = uniqueClasses.size();
    classPriorProbabilities = std::vector<double>(numOfClasses, 0);
    for (const auto& label : labels) {
        classPriorProbabilities.value()[static_cast<int>(label[0])]++;
    }
    for (auto& classPriorProbability : classPriorProbabilities.value()) {
        classPriorProbability /= labels.size();
    }
}

void GaussianNB::calculateClassMeans() {
    classMeans = std::vector<std::vector<double>>(numOfClasses, std::vector<double>(features[0].size(), 0));
    numberOfTrainingSamplesInEachClass = std::vector<uint16_t>(numOfClasses, 0);
    for (int i = 0; i < features.size(); i++) {
        int label = static_cast<int>(labels[i][0]);
        for (int j = 0; j < features[i].size(); j++) {
            classMeans[label][j] += features[i][j];
        }
        numberOfTrainingSamplesInEachClass[label]++;
    }
    for (int i = 0; i < classMeans.size(); i++) {
        for (int j = 0; j < classMeans[i].size(); j++) {
            classMeans[i][j] /= numberOfTrainingSamplesInEachClass[i];
        }
    }
}

void GaussianNB::calculateClassVariances() {
    classVariances = std::vector<std::vector<double>>(numOfClasses, std::vector<double>(features[0].size(), 0));
    for (int i = 0; i < features.size(); i++) {
        int label = static_cast<int>(labels[i][0]);
        for (int j = 0; j < features[i].size(); j++) {
            classVariances[label][j] += std::pow(features[i][j] - classMeans[label][j], 2);
        }
    }
    for (int i = 0; i < classVariances.size(); i++) {
        for (int j = 0; j < classVariances[i].size(); j++) {
            classVariances[i][j] /= numberOfTrainingSamplesInEachClass[i];
        }
    }
}

double GaussianNB::calculateLikelyhoodForFeature(const std::vector<double>& feature, const std::vector<double>& classMean, const std::vector<double>& classVariance) const {
    double exponent = -std::pow(feature[0] - classMean[0], 2) / (2 * classVariance[0]);
    double denominator = std::sqrt(2 * M_PI * classVariance[0]);
    return std::exp(exponent) / denominator;
}

double GaussianNB::calculateJointLogLikelyhood(const std::vector<double>& feature, const std::vector<double>& classMean, const std::vector<double>& classVariance) const {
    double logLikelyhood = 0;
    for (int i = 0; i < feature.size(); i++) {
        logLikelyhood += std::log(calculateLikelyhoodForFeature({ feature[i] }, { classMean[i] }, { classVariance[i] }));
    }
    return logLikelyhood;
}

std::vector<std::map<double, double>> GaussianNB::calculatePosteriorProbabilities(const std::vector<std::vector<double>>& queryData) const {
    std::vector<std::map<double, double>> probabilities;
    for (const auto& query : queryData) {
        std::map<double, double> classProbabilities;
        for (int i = 0; i < numOfClasses; i++) {
            double jointLogLikelyhood = calculateJointLogLikelyhood(query, classMeans[i], classVariances[i]);
            classProbabilities[i] = jointLogLikelyhood + std::log(classPriorProbabilities.value()[i]);
        }
        probabilities.push_back(classProbabilities);
    }
    return probabilities;
}

void GaussianNB::normalizeProbabilities(std::vector<std::map<double, double>>& probabilities) const {
    for (auto& probability : probabilities) {
        double sum = 0;
        for (const auto& [label, prob] : probability) {
            sum += std::exp(prob);
        }
        for (auto& [label, prob] : probability) {
            prob = std::exp(prob) / sum;
        }
    }
}

void GaussianNB::scaleFeatures(std::vector<std::vector<double>>& features) {
    for (int i = 0; i < features[0].size(); i++) {
        double mean = 0;
        for (int j = 0; j < features.size(); j++) {
            mean += features[j][i];
        }
        mean /= features.size();
        double variance = 0;
        for (int j = 0; j < features.size(); j++) {
            variance += std::pow(features[j][i] - mean, 2);
        }
        variance /= features.size();
        for (int j = 0; j < features.size(); j++) {
            features[j][i] = (features[j][i] - mean) / std::sqrt(variance);
        }
    }
}
