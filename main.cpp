#include <iostream>
#include "Knn.hpp"
#include "Dataset.hpp"
#include "AccuracyScore.hpp"
#include "TrainTestSplit.hpp"
#include "ModelSaver.hpp"
#include "KFold.hpp"

#include "testing.hpp"

int main() {
    // testingImplementations::testReloading();
    testingImplementations::testKFold();

    std::cout << "Still works" << std::endl;
}