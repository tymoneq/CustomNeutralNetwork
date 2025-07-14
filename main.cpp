#include "include/NeutralNetwork.h"
#include <iostream>
#include <vector>

using namespace std;

int main()
{

    vector<int> topology = {3, 2, 4, 1};

    Activation activationFunction = SIGMOID;

    NeutralNetwork MyNetwork(topology, 0.05, activationFunction);

    MyNetwork.initializeWeights();
    MyNetwork.printValues();

    vector<double> data = {5, 3, 4};

    MyNetwork.train(1, data);
    MyNetwork.printValues();

    return 0;
}