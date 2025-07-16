#include "include/NeuralNetwork.h"
#include "include/DataPreprocessing.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>

using namespace std;

void readFile(int columns, int rows, vector<vector<double>> &data, string &fileName);

int main()
{
    const double trainSize = 0.8;
    int columns = 0;
    int rows = 0;
    string fileName = "";
    vector<vector<double>> data;
    vector<int> topology = {2, 4, 1};

    // cout << "Please enter the number of columns, rows and the name of the input file\n";
    // cout << "Columns: ";
    // cin >> columns;
    // cout << "\nRow: ";
    // cin >> rows;
    // cout << "\nFile name: ";
    // cin >> fileName;
    // cout << "\n";

    readFile(columns, rows, data, fileName);

    // preprocessing
    DataSplitResult splitData = splitTestSet(data, trainSize);
    standardScaler(splitData);
    HotEncodedData encodedTypes = oneHotEncoding(splitData);

    Activation activationFunction = SIGMOID;
    NeuralNetwork MyNetwork(topology, 0.5, activationFunction);

    MyNetwork.train(40, splitData.trainX, encodedTypes.trainY);

    for (int i = 0; i < data.size(); i++)
        cout << MyNetwork.predict(data[i]) << "\n";

    return 0;
}

void readFile(int columns, int rows, vector<vector<double>> &data, string &fileName)
{

    ifstream file(fileName);

    if (!file)
        throw invalid_argument("Error opening file");

    data.resize(rows, vector<double>(columns));

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            file >> data[i][j];

    file.close();
}