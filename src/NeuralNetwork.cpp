#include "../include/NeuralNetwork.h"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace std;

NeuralNetwork::NeuralNetwork(const vector<int> &topology, double learningRate = 0.05, Activation activation = SIGMOID)
{
    this->topology = topology;
    this->learningRate = learningRate;
    this->activationFunction = activation;

    this->weights.resize(topology.size());
    this->bias.resize(topology.size());
    this->values.resize(topology.size());
    this->deltas.resize(topology.size());
}

void NeuralNetwork::initializeWeights()
{
    uniform_real_distribution<double> unif(0, 1);

    default_random_engine re;

    for (int i = 0; i < this->topology.size(); i++)
        for (int j = 0; j < this->topology[i]; j++)
        {
            this->values[i].emplace_back(0);
            this->deltas[i].emplace_back(0);

            double weight = unif(re);
            this->weights[i].emplace_back(weight);

            if (i < this->topology.size() - 1 && i > 0)
            {
                double bias = unif(re);
                this->bias[i].emplace_back(bias);
            }
            else
                this->bias[i].emplace_back(0);
        }
}

void NeuralNetwork::printValues()
{
    for (int i = 0; i < this->topology.size(); i++)
    {
        cout << "Layer: " << i << "\n";
        for (int j = 0; j < this->topology[i]; j++)
        {
            cout << this->values[i][j] << "\n";
            cout << this->deltas[i][j] << "\n";
        }

        cout << "\n";
    }
}

double NeuralNetwork::activation(double x)
{

    if (this->activationFunction == SIGMOID)
        return 1 / (1 + exp(-x));

    else if (this->activationFunction == TANH)
        return tanh(x);
}
double NeuralNetwork::activationDerivative(double x)
{
    if (this->activationFunction == SIGMOID)
        return x * (1.0 - x);
}

void NeuralNetwork::forwardPropagation()
{
    for (int layer = 1; layer < this->topology.size(); layer++)
        for (int neuron = 0; neuron < this->topology[layer]; neuron++)
        {
            double neuronValue = this->bias[layer][neuron];
            for (int prevNeuron = 0; prevNeuron < this->topology[layer - 1]; prevNeuron++)
                neuronValue += this->weights[layer - 1][prevNeuron] * this->values[layer - 1][prevNeuron];

            neuronValue = activation(neuronValue);
            this->values[layer][neuron] = neuronValue;
        }
}

void NeuralNetwork::backwardPropagateError(const vector<int> &expected)
{
    for (int layer = this->topology.size() - 1; layer > 0; layer--)
    {
        if (layer == this->topology.size() - 1)
            for (int neuron = 0; neuron < expected.size(); neuron++)
                this->deltas[layer][neuron] = (this->values[layer][neuron] - expected[neuron]) * activationDerivative(this->values[layer][neuron]);

        else
            for (int neuron = 0; neuron < topology[layer]; neuron++)
            {
                double error = 0.0;

                for (int nextNeuron = 0; nextNeuron < topology[layer + 1]; nextNeuron++)
                    error += this->weights[layer][nextNeuron] * this->deltas[layer + 1][nextNeuron] * activationDerivative(this->values[layer][neuron]);

                this->deltas[layer][neuron] = error;
            }
    }
}

void NeuralNetwork::updateWeights()
{

    for (int layer = 1; layer < this->topology.size(); layer++)
        for (int neuron = 0; neuron < this->topology[layer]; neuron++)
        {
            for (int prevNeuron = 0; prevNeuron < this->topology[layer - 1]; prevNeuron++)
                this->weights[layer - 1][prevNeuron] -= this->learningRate * this->deltas[layer][neuron] * this->values[layer - 1][prevNeuron];

            this->bias[layer][neuron] -= this->learningRate * this->deltas[layer][neuron];
        }
}

void NeuralNetwork::train(int epochs, const vector<vector<double>> &data, const vector<vector<int>> &expected)
{

    for (int i = 0; i < epochs; i++)
    {
        double sumError = 0.0;
        for (int rowData = 0; rowData < data.size(); rowData++)
        {
            if (data[rowData].size() != this->topology[0])
                throw invalid_argument("First layer has different size than the data");

            if (expected[rowData].size() != this->topology.back())
                throw invalid_argument("Output layer has different size than the data");

            for (int i = 0; i < data[rowData].size(); i++)
                this->values[0][i] = data[rowData][i];

            forwardPropagation();

            // calculating error
            int lastLayer = this->topology.back();
            for (int j = 0; j < lastLayer; j++)
                sumError += (this->values[lastLayer][j] - expected[rowData][j]) * (this->values[lastLayer][j] - expected[rowData][j]);

            backwardPropagateError(expected[rowData]);
            updateWeights();
        }
        cout << ">epoch" << i << ", lrate=" << this->learningRate << ", error" << sumError << "\n";
    }
}

NeuralNetwork::~NeuralNetwork() {}