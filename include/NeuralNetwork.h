#ifndef NEUTRAL_NETWORK_H
#define NEUTRAL_NETWORK_H

#include <vector>

using namespace std;
enum Activation
{
    SIGMOID,
    TANH
};

class NeuralNetwork
{
public:
    NeuralNetwork(const vector<int> &topology, double learningRate, Activation activation);
    ~NeuralNetwork();

    void initializeWeights();
    void printValues();
    double activation(double x);
    double activationDerivative(double x);
    void train(int epochs, const vector<vector<double>> &data, const vector<int> &expected);
    void forwardPropagation();
    void backwardPropagateError(const int &expected);
    void updateWeights();
    void updateLearningRate(int epoch);
    int predict(const vector<double> &data);

private:
    vector<vector<vector<double>>> weights;
    vector<vector<double>> bias;
    vector<vector<double>> values;
    vector<vector<double>> deltas;
    vector<int> topology;
    double learningRate;
    double initialLearningRate;
    Activation activationFunction;
};

#endif