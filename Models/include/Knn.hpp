#pragma once
#include <vector>

class Knn {
public:
    Knn(int k = 5);
    ~Knn();
    void fit(const std::vector<std::vector<double>>& features, const std::vector<std::vector<double>>& labels);
    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& queryData) const;
private:
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> labels;
    int k;

    std::vector<double> getSortedEuclideanDistances(const std::vector<double>& queryData) const;
    std::vector<double> getLabelsForFeature(const std::vector<double>& feature, std::vector<double>& distances) const;

};
