#include "include/NeuralNetwork.h"
#include <iostream>
#include <vector>

using namespace std;

int main()
{

    vector<int> topology = {3, 2, 4, 1};

    Activation activationFunction = SIGMOID;

    NeuralNetwork MyNetwork(topology, 0.05, activationFunction);

    MyNetwork.initializeWeights();


    vector<vector<double>> data = {{5, 3, 4}};
    vector<vector<int>> expected = {{1}};

    MyNetwork.train(100, data, expected);


    return 0;
}