#include "Dataset.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

Dataset::Dataset(int numLabels, bool headerExists, char delimiter) : numLabels(numLabels), headerExists(headerExists), delimiter(delimiter) {}

Dataset::~Dataset() {}

// Assumes the data is in the format of features followed by labels
// Feautre1, Feature2, ..., FeatureN, Label1, Label2, ..., LabelM

void Dataset::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }
    std::string line;
    if (headerExists) {
        std::getline(file, line);
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, delimiter)) {
            if (numLabels > 0) {
                if (labelNames.size() < numLabels) {
                    labelNames.push_back(token);
                } else {
                    featureNames.push_back(token);
                }
            } else {
                featureNames.push_back(token);
            }
        }
    }
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> feature;
        std::vector<double> label;
        int count = 0;
        while (std::getline(ss, token, delimiter)) {
            if (count < featureNames.size()) {
                feature.push_back(std::stod(token));
            } else {
                label.push_back(std::stod(token));
            }
            count++;
        }
        features.push_back(feature);
        labels.push_back(label);
    }
}

std::vector<std::vector<double>> Dataset::getFeatures() const {
    return features;
}

std::vector<std::vector<double>> Dataset::getLabels() const {
    return labels;
}

std::vector<std::string> Dataset::getLabelNames() const {
    if (headerExists) {
        return labelNames;
    } else {
        //return empty vector
        return std::vector<std::string>();
    }
}

std::vector<std::string> Dataset::getFeatureNames() const {
    if (headerExists) {
        return featureNames;
    } else {
        //return empty vector
        return std::vector<std::string>();
    }
}
