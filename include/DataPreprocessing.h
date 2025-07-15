#ifndef DATA_PREPROCESSING
#define DATA_PREPROCESSING

#include <vector>
#include <algorithm>
#include <random>

using namespace std;

struct DataSplitResult
{
    vector<vector<double>> trainX, testX;
    vector<int> trainY, testY;
};

DataSplitResult splitTestSet(vector<vector<double>> &data, double trainSize);

#endif DATA_PREPROCESSING