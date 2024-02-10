#pragma once
#include "BaseModel.hpp"
class Knn : public BaseModel {
public:
    Knn(int k = 5);
    ~Knn();
    void fit(const std::vector<std::vector<double>>& features, const std::vector<std::vector<double>>& labels) override;
    void partialFit(const std::vector<std::vector<double>>& features, const std::vector<std::vector<double>>& labels) override;
    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& queryData) const override;
    std::vector<std::map<double, double>> predictProba(const std::vector<std::vector<double>>& queryData) const override;

    // Serialize function for boost
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar& features;
        ar& labels;
        ar& k;
    }

private:
    int k;
    std::vector<double> getSortedEuclideanDistances(const std::vector<double>& queryData) const;
    std::vector<double> getLabelsForFeature(const std::vector<double>& feature, std::vector<double>& distances) const;

};
