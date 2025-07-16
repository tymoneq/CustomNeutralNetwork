#include "include/NeuralNetwork.h"
#include "include/DataPreprocessing.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
#include <iomanip>
using namespace std;

void readFile(int columns, int rows, vector<vector<double>> &data, string &fileName);

void calculateAccuracy(NeuralNetwork &MyNetwork, DataSplitResult &testX, HotEncodedData &testY);
int main()
{
    const double trainSize = 0.8;
    int columns = 8;
    int rows = 210;
    string fileName = "seeds_dataset.txt";
    vector<vector<double>> data;
    vector<int> topology = {7, 8, 3};

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

    MyNetwork.train(500, splitData.trainX, encodedTypes.trainY);

    calculateAccuracy(MyNetwork, splitData, encodedTypes);

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

void calculateAccuracy(NeuralNetwork &MyNetwork, DataSplitResult &testX, HotEncodedData &testY)
{

    int correct = 0;
    for (int i = 0; i < testX.testX.size(); i++)
    {
        int answer = 0;
        for (int j = 0; j < testY.testY[i].size(); j++)
            if (testY.testY[i][j] == 1)
                answer = j;

        int predictedAnswer = MyNetwork.predict(testX.testX[i]);

        if (predictedAnswer == answer)
            correct++;
    }

    double accuracy = (double)correct / (double)testX.testX.size() * 100;

    cout << fixed << setprecision(2) << "Model has an accuracy of " << accuracy << "% \n";
}