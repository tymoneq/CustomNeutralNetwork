#ifndef NEUTRAL_NETWORK_H
#define NEUTRAL_NETWORK_H

#include <vector>

using namespace std;

class NeutralNetwork
{
public:
    enum Activation
    {
        SIGMOID,
        TANH,
        RELU,
        ELU,
        SELU,
        SWISH,
    };
    NeutralNetwork(const vector<int> &topology, double learningRate, Activation activation);
    ~NeutralNetwork();

    void initialize_weights();
    void print_weights();
    double activation(double x);
    void train(int epochs);

private:
    vector<vector<double>> weights;
    vector<vector<double>> bias;
    vector<vector<double>> values;
    vector<int> topology;
    double learningRate;
    Activation activationFunction;
};

#endif