#include "../include/NeutralNetwork.h"
#include <cmath>
#include <iostream>
#include <random>

using namespace std;

NeutralNetwork::NeutralNetwork(const vector<int> &topology, double learningRate = 0.05, Activation activation = SIGMOID)
{
    this->topology = topology;
    this->learningRate = learningRate;
    this->activationFunction = activation;

    this->weights.resize(topology.size());
    this->bias.resize(topology.size());
    this->values.resize(topology.size());
}

void NeutralNetwork::initialize_weights()
{
    uniform_real_distribution<double> unif(0, 1);

    default_random_engine re;

    for (int i = 0; i < this->topology.size(); i++)
        for (int j = 0; j < this->topology[i]; j++)
        {
            double weight = unif(re);
            this->weights[i].emplace_back(weight);
            this->values[i].emplace_back(0);

            if (i < this->topology.size() - 1 && i > 0)
            {
                double bias = unif(re);
                this->bias[i].emplace_back(bias);
            }
        }
}

void NeutralNetwork::print_weights()
{
    for (int i = 0; i < this->topology.size(); i++)
    {
        cout << "Layer: " << i << "\n";
        for (int j = 0; j < this->topology[i]; j++)
            cout << this->weights[i][j] << " ";

        cout << "\n";
    }
}

double NeutralNetwork::activation(double x)
{
    const double alpha = 1.6733;
    const double lambda = 1.0507;

    if (this->activationFunction == SIGMOID)
        return 1 / (1 + exp(-x));

    else if (this->activationFunction == TANH)
        return tanh(x);

    else if (this->activationFunction == RELU)
        return max((double)0, x);

    else if (this->activationFunction == ELU)
    {
        if (x <= 0)
            return alpha * (exp(x) - 1);

        else
            return x;
    }
    else if (this->activationFunction == SELU)
    {
        if (x <= 0)
            return lambda * alpha * (exp(x) - 1);
        else
            return lambda * x;
    }
    else if (this->activationFunction == SWISH)
        return x / (1 + exp(-x));
    
}

NeutralNetwork::~NeutralNetwork() {}
