#pragma once
#include <vector>
#include <string>

class Dataset {
public:
    Dataset(int numLabels, bool headerExists = true, char delimiter = ',');
    Dataset(int numLabels, std::vector<std::vector<double>> features,
        std::vector<std::vector<double>> labels, std::vector<std::string> labelNames, std::vector<std::string> featureNames, bool headerExists = true, char delimiter = ',');
    
    ~Dataset();
    Dataset(const Dataset& dataset);
    Dataset& operator=(const Dataset& dataset);
    Dataset(Dataset&& dataset);
    Dataset& operator=(Dataset&& dataset);

    void load(const std::string& filename);

    std::vector<std::vector<double>> getFeatures() const;
    std::vector<std::vector<double>> getLabels() const;
    std::vector<std::string> getLabelNames() const;
    std::vector<std::string> getFeatureNames() const;
    int getNumLabels() const;
    bool getHeaderExists() const;
    char getDelimiter() const;

    // Serialize function for boost
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar& features;
        ar& labels;
        ar& labelNames;
        ar& featureNames;
        ar& numLabels;
        ar& headerExists;
        ar& delimiter;
    }
private:
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> labels;
    std::vector<std::string> labelNames;
    std::vector<std::string> featureNames;
    int numLabels;
    bool headerExists;
    char delimiter;
};
