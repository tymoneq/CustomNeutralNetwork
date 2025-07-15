#include "include/NeuralNetwork.h"
#include <iostream>
#include <vector>

using namespace std;

int main()
{

    vector<int> topology = {2, 4, 1};

    Activation activationFunction = SIGMOID;

    NeuralNetwork MyNetwork(topology, 0.5, activationFunction);

    MyNetwork.initializeWeights();

    vector<vector<double>> data = {
        {7.673756466, 3.508563011},
        {1.465489372, 2.362125076},
        {3.396561688, 4.400293529},
        {1.38807019, 1.850220317},
        {3.06407232, 3.005305973},
        {7.627531214, 2.759262235},
        {5.332441248, 2.088626775},
        {6.922596716, 1.77106367},
        {8.675418651, -0.242068655},
        {2.7810836, 2.550537003}};

    vector<int> expected = {1, 0, 0, 0, 0, 1, 1, 1, 1, 0};

    MyNetwork.train(40, data, expected);

    for (int i = 0; i < data.size(); i++)
        cout << MyNetwork.predict(data[i]) << "\n";

    return 0;
}