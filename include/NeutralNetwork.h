#ifndef NEUTRAL_NETWORK_H
#define NEUTRAL_NETWORK_H

#include <vector>

using namespace std;
enum Activation
{
    SIGMOID,
    TANH,
    RELU,
    ELU,
    SELU,
    SWISH,
};
class NeutralNetwork
{
public:
    NeutralNetwork(const vector<int> &topology, double learningRate, Activation activation);
    ~NeutralNetwork();

    void initializeWeights();
    void printValues();
    double activation(double x);
    void train(int epochs, const vector<double> &data);
    void forwardPropagation();

private:
    vector<vector<double>> weights;
    vector<vector<double>> bias;
    vector<vector<double>> values;
    vector<int> topology;
    double learningRate;
    Activation activationFunction;
};

#endif