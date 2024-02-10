#include "Knn.hpp"

#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
Knn::Knn(int k) : k(k) {}

Knn::~Knn() {}

void Knn::fit(const std::vector<std::vector<double>>& features, const std::vector<std::vector<double>>& labels) {
    this->features = features;
    this->labels = labels;
}

void Knn::partialFit(const std::vector<std::vector<double>>& features, const std::vector<std::vector<double>>& labels) {
    this->features.insert(this->features.end(), features.begin(), features.end());
    this->labels.insert(this->labels.end(), labels.begin(), labels.end());
}

std::vector<std::vector<double>> Knn::predict(const std::vector<std::vector<double>>& queryData) const {
    std::vector<std::vector<double>> predictions;
    for (const auto& query : queryData) {
        std::vector<double> distances = getSortedEuclideanDistances(query);
        std::vector<double> labels = getLabelsForFeature(query, distances);
        predictions.push_back(labels);
    }
    return predictions;
}


std::vector<std::map<double, double>> Knn::predictProba(const std::vector<std::vector<double>>& queryData) const {
    // ToDo: Confirm if this is the correct implementation
    std::vector<std::map<double, double>> predictions;
    for (const auto& query : queryData) {
        std::vector<double> distances = getSortedEuclideanDistances(query);

        std::map<double, int> labelCounts;

        for (int i = 0; i < k && i < distances.size(); i++) {

            double label = labels[i][0]; // Simplified; might need adjustments 
            labelCounts[label]++;
        }
        // Convert counts to probabilities by dividing by k
        std::map<double, double> labelProbabilities;
        for (const auto& labelCount : labelCounts) {
            labelProbabilities[labelCount.first] = static_cast<double>(labelCount.second) / k;
        }

        predictions.push_back(labelProbabilities);
    }

    return predictions;
}


std::vector<double> Knn::getSortedEuclideanDistances(const std::vector<double>& queryData) const {
    std::vector<double> distances;
    for (const auto& feature : features) {
        double distance = 0;
        for (int i = 0; i < feature.size(); i++) {
            distance += std::pow(feature[i] - queryData[i], 2);
        }
        distances.push_back(std::sqrt(distance));
    }
    std::sort(distances.begin(), distances.end());
    return distances;
}

// Taking the average of the labels of the k nearest neighbors
std::vector<double> Knn::getLabelsForFeature(const std::vector<double>& feature, std::vector<double>& distances) const {
    std::map<double, double> labelCount;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < features.size(); j++) {
            if (distances[i] == std::sqrt(std::pow(features[j][0] - feature[0], 2) + std::pow(features[j][1] - feature[1], 2))) {
                labelCount[labels[j][0]]++;
            }
        }
    }
    std::vector<double> labels;
    for (const auto& pair : labelCount) {
        labels.push_back(pair.first);
    }
    return labels;
}
