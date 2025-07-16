#include "../include/DataPreprocessing.h"
#include <vector>
#include <algorithm>
#include <random>
#include <set>
using namespace std;

DataSplitResult splitTestSet(vector<vector<double>> &data, double trainSize)
{

    random_device rd;
    mt19937 g(rd());
    shuffle(data.begin(), data.end(), g);

    vector<int> classType;
    vector<vector<double>> dataLabels(data.size());
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

void scaler(vector<vector<double>> &data)
{
    for (int col = 0; col < data.size(); col++)
    {
        double mean = 0.0;
        double standardDeviation = 0.0;
        for (int row = 0; row < data[col].size(); row++)
            mean += data[col][row];

        mean /= (double)data[col].size();

        for (int row = 0; row < data[col].size(); row++)
            standardDeviation += pow((double)data[col][row] - mean, 2);

        standardDeviation = sqrt(standardDeviation / data[col].size());

        for (int row = 0; row < data[col].size(); row++)
            data[col][row] = (data[col][row] - mean) / standardDeviation;
    }
}

void standardScaler(DataSplitResult &data)
{

    scaler(data.trainX);
    scaler(data.testX);
}

vector<vector<int>> encoder(vector<int> &data)
{
    set<int> types;

    for (auto w : data)
        types.insert(w);

    vector<vector<int>> encodedResults(data.size(), vector<int>(types.size(), 0));

    for (int i = 0; i < data.size(); i++)
        encodedResults[i][data[i] - 1] = 1;

    return encodedResults;
}

HotEncodedData oneHotEncoding(DataSplitResult &data)
{

    HotEncodedData encodedData;

    encodedData.trainY = encoder(data.trainY);
    encodedData.testY = encoder(data.testY);

    return encodedData;
}