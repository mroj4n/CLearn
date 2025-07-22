Machine learning models implementation using *OOP* principles in C++ 


Sample code usage: 
```c++
void knnTesting()
{
    AccuracyScore accuracyScore;

    //create dataset object with number of labels and load dataset
    Dataset dataset(1);
    dataset.load("../DataExamples/classification_data.csv");

    //split the dataset into training and testing dataset
    TrainTestSplit trainTestSplit(0.2, dataset, 42);
    Dataset trainDataset = trainTestSplit.getTrainDataset();
    Dataset testDataset = trainTestSplit.getTestDataset();


    //initialize classifier
    //in case of knn give number of neighbors
    Knn knn(3);

    //train the model
    knn.fit(trainDataset.getFeatures(), trainDataset.getLabels());

    //prediction
    std::vector<std::vector<double>> predictions = knn.predict(testDataset.getFeatures());
    std::cout << "Accuracy: " << accuracyScore.calculate(predictions, testDataset.getLabels()) << std::endl;

    // Save and reload model and recheck accuracy
    Utils::ModelSaver::saveModel(knn, "knn_model");
    Knn knnReloaded = Utils::ModelSaver::loadModel<Knn>("knn_model");
    std::vector<std::vector<double>> predictionsReloaded = knnReloaded.predict(testDataset.getFeatures());
    std::cout << "Accuracy reloaded: " << accuracyScore.calculate(predictionsReloaded, testDataset.getLabels()) << std::endl;
}
```

Availaible models 
- KNN
- GaussianNB

Availaible Metrics
- AccuracyScore

Availiable Validators
- KFold Cross validator

Some other examples can be found in `Utils/include/testing.hpp`
