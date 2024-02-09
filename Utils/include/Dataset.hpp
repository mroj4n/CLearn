#pragma once
#include <vector>
#include <string>

class Dataset {
public:
    Dataset(int numLabels, bool headerExists = true, char delimiter = ',');
    ~Dataset();
    void load(const std::string& filename);
    std::vector<std::vector<double>> getFeatures() const;
    std::vector<std::vector<double>> getLabels() const;
    std::vector<std::string> getLabelNames() const;
    std::vector<std::string> getFeatureNames() const;
private:
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> labels;
    std::vector<std::string> labelNames;
    std::vector<std::string> featureNames;
    int numLabels;
    bool headerExists;
    char delimiter;
};
