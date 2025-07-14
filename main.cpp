#include "include/NeutralNetwork.h"
#include <iostream>
#include <vector>

using namespace std;

int main()
{

    vector<int> topology = {3, 2, 4, 1};

    NeutralNetwork MyNetwork(topology, 0.05, SIGMOID);

    MyNetwork.initialize_weights();
    MyNetwork.print_weights();

    return 0;
}