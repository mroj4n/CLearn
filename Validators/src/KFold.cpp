#include "KFold.hpp"
#include "DataManipulation.hpp"
#include <iostream>
#include <numeric>
KFold::KFold(const Dataset& dataset, const uint16_t& numOfFolds, std::optional<int> randomSeed) : dataset(dataset), numOfFolds(numOfFolds)
{
    uint16_t datasetSize = dataset.getFeatures().size();
    if (numOfFolds > datasetSize)
        throw std::invalid_argument("numOfFolds cannot be greater than the dataset size");


    std::vector<uint16_t> indices;
    for (uint16_t i = 0; i < datasetSize; i++)
    {
        indices.push_back(i);
    }
    if (randomSeed.has_value())
        indices = DataManipulation::randomizeIndexes(indices, randomSeed);

    uint16_t foldSize = datasetSize / numOfFolds;
    uint16_t remainder = datasetSize % numOfFolds;

    uint16_t current_indices_index = 0;
    for (uint16_t i = 0; i < numOfFolds; i++)
    {
        std::vector<uint16_t> foldIndices;
        for (uint16_t j = 0; j < foldSize; j++)
        {
            foldIndices.push_back(indices[current_indices_index++]);
        }
        if (remainder > 0)
        {
            foldIndices.push_back(indices[current_indices_index++]);
            remainder--;
        }
        indicesForEachFold.push_back(foldIndices);
    }

}

KFold::~KFold()
{
}

std::vector<std::vector<uint16_t>> KFold::getIndices() const
{
    return indicesForEachFold;
}

std::vector<std::pair<Dataset,Dataset>> KFold::getDatasetsForEachFold() const
{
    std::vector<std::pair<Dataset, Dataset>> datasets;
    for (uint16_t i = 0; i < indicesForEachFold.size(); i++)
    {
        std::vector<std::vector<double>> trainFeatures;
        std::vector<std::vector<double>> trainLabels;
        std::vector<std::vector<double>> testFeatures;
        std::vector<std::vector<double>> testLabels;
        std::vector<std::string> labelNames = dataset.getLabelNames();
        std::vector<std::string> featureNames = dataset.getFeatureNames();
        for (uint16_t j = 0; j < indicesForEachFold.size(); j++)
        {
            if (j != i)
            {
                for (auto index : indicesForEachFold[j])
                {
                    trainFeatures.push_back(dataset.getFeatures()[index]);
                    trainLabels.push_back(dataset.getLabels()[index]);
                }
            }
            else
            {
                for (auto index : indicesForEachFold[j])
                {
                    testFeatures.push_back(dataset.getFeatures()[index]);
                    testLabels.push_back(dataset.getLabels()[index]);
                }
            }
        }
        Dataset trainDataset(dataset.getNumLabels(), trainFeatures, trainLabels, labelNames, featureNames, dataset.getHeaderExists(), dataset.getDelimiter());
        Dataset testDataset(dataset.getNumLabels(), testFeatures, testLabels, labelNames, featureNames, dataset.getHeaderExists(), dataset.getDelimiter());
        datasets.push_back(std::make_pair(trainDataset, testDataset));
    }
    return datasets;
}

