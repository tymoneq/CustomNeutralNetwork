#include "../include/DataPreprocessing.h"
#include <vector>
#include <algorithm>
#include <random>

using namespace std;

DataSplitResult splitTestSet(vector<vector<double>> &data, double trainSize)
{

    random_device rd;
    mt19937 g(rd());
    shuffle(data.begin(), data.end(), g);

    vector<int> classType;
    vector<vector<double>> dataLabels;
    for (int i = 0; i < data.size(); i++)
    {
        classType.emplace_back(data[i][data[i].size() - 1]);
        dataLabels[i].resize(data[i].size() - 1);

        for (int j = 0; j < data[i].size() - 1; j++)
            dataLabels[i][j] = data[i][j];
    }

    // spliting the data
    int splitIndex = static_cast<int>(data.size() * trainSize);

    vector<vector<double>> trainX(dataLabels.begin(), dataLabels.begin() + splitIndex);
    vector<vector<double>> testX(dataLabels.begin() + splitIndex, dataLabels.end());
    vector<int> trainY(classType.begin(), classType.begin() + splitIndex);
    vector<int> testY(classType.begin() + splitIndex, classType.end());

    return {trainX, testX, trainY, testY};
}