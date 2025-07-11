#include <eigen3/Eigen/Eigen>
#include <vector>
#include <iostream>

using namespace std;

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

class NeutralNetwork
{
protected:
    vector<RowVector *> neuronLayers;
    vector<RowVector *> cacheLayers;
    vector<RowVector *> deltas;
    vector<Matrix *> weights;
    vector<RowVector *> biases;
    Scalar learningRate;
    vector<uint> topology;

public:
    NeutralNetwork(vector<uint> topology, Scalar learningRate = Scalar(0.005))
    {
        this->topology = topology;
        this->learningRate = learningRate;

        // creating neural network from topology and adding extra bias neuron to all layers except the last one 
        for (uint i = 0; i < topology.size(); i++)
        {
            if (i == topology.size() - 1)
                neuronLayers.push_back(new RowVector(topology[i]));

            else
                neuronLayers.push_back(new RowVector(topology[i] + 1));
        }
    }
};

int main()
{

    return 0;
}