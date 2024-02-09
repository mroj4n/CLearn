#include "AccuracyScore.hpp"

double AccuracyScore::calculate(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& actual){
    int correct = 0;
    for (int i = 0; i < predictions.size(); i++) {
        if (predictions[i][0] == actual[i][0]) {
            correct++;
        }
    }
    return (double) correct / predictions.size();
}