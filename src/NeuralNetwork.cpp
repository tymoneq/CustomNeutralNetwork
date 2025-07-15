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
    this->initialLearningRate = learningRate;

    this->weights.resize(topology.size());
    this->bias.resize(topology.size());
    this->values.resize(topology.size());
    this->deltas.resize(topology.size());
}

void NeuralNetwork::initializeWeights()
{
    uniform_real_distribution<double> unif(-1.0, 1.0);

    default_random_engine re;

    for (int i = 0; i < this->topology.size(); i++)
    {
        this->bias[i].resize(topology[i], 0);
        this->deltas[i].resize(topology[i], 0);
        this->values[i].resize(topology[i], 0);
        this->weights[i].resize(topology[i]);

        for (int j = 0; j < this->topology[i]; j++)
        {
            if (i < this->topology.size() - 1)
            {
                this->weights[i][j].resize(this->topology[i + 1]);

                for (int k = 0; k < this->topology[i + 1]; k++)
                    weights[i][j][k] = unif(re);
            }
            if (i < this->topology.size() - 1 && i > 0)
                this->bias[i][j] = unif(re);
        }
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

    else if (this->activationFunction == TANH)
        return 1.0 - (x * x);
}

void NeuralNetwork::forwardPropagation()
{
    for (int layer = 1; layer < this->topology.size(); layer++)
        for (int neuron = 0; neuron < this->topology[layer]; neuron++)
        {
            double neuronValue = this->bias[layer][neuron];
            for (int prevNeuron = 0; prevNeuron < this->topology[layer - 1]; prevNeuron++)
                neuronValue += this->weights[layer - 1][prevNeuron][neuron] * this->values[layer - 1][prevNeuron];

            neuronValue = activation(neuronValue);
            this->values[layer][neuron] = neuronValue;
        }
}

void NeuralNetwork::backwardPropagateError(const int &expected)
{
    for (int layer = this->topology.size() - 1; layer > 0; layer--)
    {
        if (layer == this->topology.size() - 1)
            for (int neuron = 0; neuron < this->topology.back(); neuron++)
                this->deltas[layer][neuron] = (this->values[layer][neuron] - expected) * activationDerivative(this->values[layer][neuron]);

        else
            for (int neuron = 0; neuron < topology[layer]; neuron++)
            {
                double error = 0.0;

                for (int nextNeuron = 0; nextNeuron < topology[layer + 1]; nextNeuron++)
                    error += this->weights[layer][neuron][nextNeuron] * this->deltas[layer + 1][nextNeuron];

                this->deltas[layer][neuron] = error * activationDerivative(this->values[layer][neuron]);
            }
    }
}

void NeuralNetwork::updateWeights()
{

    for (int layer = 1; layer < this->topology.size(); layer++)
        for (int neuron = 0; neuron < this->topology[layer]; neuron++)
        {
            for (int prevNeuron = 0; prevNeuron < this->topology[layer - 1]; prevNeuron++)
                this->weights[layer - 1][prevNeuron][neuron] -= this->learningRate * this->deltas[layer][neuron] * this->values[layer - 1][prevNeuron];

            this->bias[layer][neuron] -= this->learningRate * this->deltas[layer][neuron];
        }
}

void NeuralNetwork::updateLearningRate(int epoch)
{
    this->learningRate = this->initialLearningRate * exp(-0.01 * epoch);
}

int NeuralNetwork::predict(const vector<double> &data)
{

    for (int i = 0; i < data.size(); i++)
        this->values[0][i] = data[i];
    forwardPropagation();

    int answer = 0;
     double maxVal = this->values[this->topology.size() - 1][0];

    if (topology.back() > 1)
    {
        for (int i = 0; i < this->topology.back(); i++)
            if (maxVal < this->values[this->topology.size() - 1][i])
            {
                maxVal = this->values[this->topology.size() - 1][i];
                answer = i;
            }
    }

    else
    {
        if (this->values[topology.size() - 1][0] >= 0.5)
            answer = 1;
        else
            answer = 0;
    }

    return answer;
}

void NeuralNetwork::train(int epochs, const vector<vector<double>> &data, const vector<int> &expected)
{

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double sumError = 0.0;
        for (int rowData = 0; rowData < data.size(); rowData++)
        {
            if (data[rowData].size() != this->topology[0])
                throw invalid_argument("First layer has different size than the data");

            for (int i = 0; i < data[rowData].size(); i++)
                this->values[0][i] = data[rowData][i];

            forwardPropagation();

            // calculating error
            int lastLayer = this->topology.size() - 1;
            for (int j = 0; j < this->topology.back(); j++)
                sumError += (this->values[lastLayer][j] - expected[rowData]) * (this->values[lastLayer][j] - expected[rowData]);

            backwardPropagateError(expected[rowData]);
            updateWeights();
        }
        updateLearningRate(epoch);
        cout << ">epoch" << epoch << ", lrate=" << this->learningRate << ", error" << sumError << "\n";
    }
}

NeuralNetwork::~NeuralNetwork() {}