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
struct HotEncodedData
{
    vector<vector<int>> trainY, testY;
};

DataSplitResult splitTestSet(vector<vector<double>> &data, double trainSize);

void scaler(vector<vector<double>> &data);
void standardScaler(DataSplitResult &data);

vector<vector<int>> encoder(vector<int> &data);
HotEncodedData oneHotEncoding(DataSplitResult &data);

#endif