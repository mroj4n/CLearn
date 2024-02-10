#include "Dataset.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
Dataset::Dataset(int numLabels, bool headerExists, char delimiter) : numLabels(numLabels), headerExists(headerExists), delimiter(delimiter) {
    setNumberOfUniqueClasses();
}

Dataset::Dataset(int numLabels, std::vector<std::vector<double>> features, std::vector<std::vector<double>> labels,
    std::vector<std::string> labelNames, std::vector<std::string> featureNames, uint16_t numOfClasses, bool headerExists, char delimiter) :
    numLabels(numLabels), features(features), labels(labels), labelNames(labelNames), featureNames(featureNames), headerExists(headerExists), delimiter(delimiter), numOfClasses(numOfClasses) {
    }

Dataset::~Dataset() {}

Dataset::Dataset(const Dataset& dataset) {
    numLabels = dataset.numLabels;
    features = dataset.features;
    labels = dataset.labels;
    labelNames = dataset.labelNames;
    featureNames = dataset.featureNames;
    headerExists = dataset.headerExists;
    delimiter = dataset.delimiter;
    numOfClasses = dataset.numOfClasses;
}

Dataset& Dataset::operator=(const Dataset& dataset) {
    if (this == &dataset) {
        return *this;
    }
    numLabels = dataset.numLabels;
    features = dataset.features;
    labels = dataset.labels;
    labelNames = dataset.labelNames;
    featureNames = dataset.featureNames;
    headerExists = dataset.headerExists;
    delimiter = dataset.delimiter;
    numOfClasses = dataset.numOfClasses;
    return *this;
}

Dataset::Dataset(Dataset&& dataset) {
    numLabels = dataset.numLabels;
    features = std::move(dataset.features);
    labels = std::move(dataset.labels);
    labelNames = std::move(dataset.labelNames);
    featureNames = std::move(dataset.featureNames);
    headerExists = dataset.headerExists;
    delimiter = dataset.delimiter;
    numOfClasses = dataset.numOfClasses;
}

Dataset& Dataset::operator=(Dataset&& dataset) {
    if (this == &dataset) {
        return *this;
    }
    numLabels = dataset.numLabels;
    features = std::move(dataset.features);
    labels = std::move(dataset.labels);
    labelNames = std::move(dataset.labelNames);
    featureNames = std::move(dataset.featureNames);
    headerExists = dataset.headerExists;
    delimiter = dataset.delimiter;
    numOfClasses = dataset.numOfClasses;
    return *this;
}

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
                }
                else {
                    featureNames.push_back(token);
                }
            }
            else {
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
            }
            else {
                label.push_back(std::stod(token));
            }
            count++;
        }
        features.push_back(feature);
        labels.push_back(label);
    }
}

void Dataset::setNumberOfUniqueClasses() {
    std::vector<double> uniqueClasses;
    for (const auto& label : labels) {
        if (std::find(uniqueClasses.begin(), uniqueClasses.end(), label[0]) == uniqueClasses.end()) {
            uniqueClasses.push_back(label[0]);
        }
    }
    numOfClasses = uniqueClasses.size();
}

uint16_t Dataset::getNumberOfUniqueClasses() const {
    return numOfClasses;
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
    }
    else {
        //return empty vector
        return std::vector<std::string>();
    }
}

std::vector<std::string> Dataset::getFeatureNames() const {
    if (headerExists) {
        return featureNames;
    }
    else {
        //return empty vector
        return std::vector<std::string>();
    }
}

int Dataset::getNumLabels() const {
    return numLabels;
}

bool Dataset::getHeaderExists() const {
    return headerExists;
}

char Dataset::getDelimiter() const {
    return delimiter;
}

