using DeveMachineLearning.ML.Functions;
using System;
using System.Collections.Generic;
using System.Text;

namespace DeveMachineLearning.ML
{
    public static class NetworkBuilder
    {
        public static List<List<Node>> BuildNetwork(
            List<int> networkShape,
            IActivationFunction activation,
            IActivationFunction outputActivation,
            IRegularizationFunction regularization,
            List<string> inputIds,
            bool initZero)
        {
            var numLayers = networkShape.Count;
            var id = 1;
            // List of layers, with each layer being a list of nodes.
            var network = new List<List<Node>>();
            for (int layerIdx = 0; layerIdx < numLayers; layerIdx++)
            {
                var isOutputLayer = layerIdx == numLayers - 1;
                var isInputLayer = layerIdx == 0;
                var currentLayer = new List<Node>();
                network.Add(currentLayer);
                var numNodes = networkShape[layerIdx];
                for (int i = 0; i < numNodes; i++)
                {
                    var nodeId = id.ToString();
                    if (isInputLayer)
                    {
                        nodeId = inputIds[i];
                    }
                    else
                    {
                        id++;
                    }
                    var node = new Node(nodeId, isOutputLayer ? outputActivation : activation, initZero);
                    currentLayer.Add(node);
                    if (layerIdx >= 1)
                    {
                        for (int j = 0; j < network[layerIdx - 1].Count; j++)
                        {
                            var prevNode = network[layerIdx - 1][j];
                            var link = new Link(prevNode, node, regularization, initZero);
                            prevNode.Outputs.Add(link);
                            node.InputLinks.Add(link);
                        }
                    }
                }
            }
            return network;
        }

        public static double ForwardProp(List<List<Node>> network, List<double> inputs)
        {
            var inputLayer = network[0];
            if (inputs.Count != inputLayer.Count)
            {
                throw new InvalidOperationException("The number of inputs must match the number of nodes in the input layer");
            }
            // Update the input layer.
            for (int i = 0; i < inputLayer.Count; i++)
            {
                var node = inputLayer[i];
                node.Output = inputs[i];
            }
            for (int layerIdx = 1; layerIdx < network.Count; layerIdx++)
            {
                var currentLayer = network[layerIdx];
                // Update all the nodes in this layer.
                for (var i = 0; i < currentLayer.Count; i++)
                {
                    var node = currentLayer[i];
                    node.UpdateOutput();
                }
            }
            return network[network.Count - 1][0].Output;
        }

        public static void BackProp(List<List<Node>> network, double target, IErrorFunction errorFunc)
        {
            // The output node is a special case. We use the user-defined error
            // function for the derivative.
            var outputNode = network[network.Count - 1][0];
            outputNode.OutputDer = errorFunc.Der(outputNode.Output, target);

            // Go through the layers backwards.
            for (var layerIdx = network.Count - 1; layerIdx >= 1; layerIdx--)
            {
                var currentLayer = network[layerIdx];
                // Compute the error derivative of each node with respect to:
                // 1) its total input
                // 2) each of its input weights.
                for (var i = 0; i < currentLayer.Count; i++)
                {
                    var node = currentLayer[i];
                    node.InputDer = node.OutputDer * node.ActivationFunction.Der(node.TotalInput);
                    node.AccInputDer += node.InputDer;
                    node.NumAccumulatedDers++;
                }

                // Error derivative with respect to each weight coming into the node.
                for (var i = 0; i < currentLayer.Count; i++)
                {
                    var node = currentLayer[i];
                    for (var j = 0; j < node.InputLinks.Count; j++)
                    {
                        var link = node.InputLinks[j];
                        if (link.IsDead)
                        {
                            continue;
                        }
                        link.ErrorDer = node.InputDer * link.Source.Output;
                        link.AccErrorDer += link.ErrorDer;
                        link.NumAccumulatedDers++;
                    }
                }
                if (layerIdx == 1)
                {
                    continue;
                }
                var prevLayer = network[layerIdx - 1];
                for (var i = 0; i < prevLayer.Count; i++)
                {
                    var node = prevLayer[i];
                    // Compute the error derivative with respect to each node's output.
                    node.OutputDer = 0;
                    for (var j = 0; j < node.Outputs.Count; j++)
                    {
                        var output = node.Outputs[j];
                        node.OutputDer += output.Weight * output.Dest.InputDer;
                    }
                }
            }
        }

        public static void UpdateWeights(List<List<Node>> network, double learningRate, double regularizationRate)
        {
            for (var layerIdx = 1; layerIdx < network.Count; layerIdx++)
            {
                var currentLayer = network[layerIdx];
                for (var i = 0; i < currentLayer.Count; i++)
                {
                    var node = currentLayer[i];
                    // Update the node's bias.
                    if (node.NumAccumulatedDers > 0)
                    {
                        node.Bias -= learningRate * node.AccInputDer / node.NumAccumulatedDers;
                        node.AccInputDer = 0;
                        node.NumAccumulatedDers = 0;
                    }
                    // Update the weights coming into this node.
                    for (var j = 0; j < node.InputLinks.Count; j++)
                    {
                        var link = node.InputLinks[j];
                        if (link.IsDead)
                        {
                            continue;
                        }
                        var regulDer = link.RegularizationFunction != null ? link.RegularizationFunction.Der(link.Weight) : 0;
                        if (link.NumAccumulatedDers > 0)
                        {
                            // Update the weight based on dE/dw.
                            link.Weight = link.Weight -
                                (learningRate / link.NumAccumulatedDers) * link.AccErrorDer;
                            // Further update the weight based on regularization.
                            var newLinkWeight = link.Weight -
                                (learningRate * regularizationRate) * regulDer;
                            if (link.RegularizationFunction == RegularizationFunction.L1 &&
                                link.Weight * newLinkWeight < 0)
                            {
                                // The weight crossed 0 due to the regularization term. Set it to 0.
                                link.Weight = 0;
                                link.IsDead = true;
                            }
                            else
                            {
                                link.Weight = newLinkWeight;
                            }
                            link.AccErrorDer = 0;
                            link.NumAccumulatedDers = 0;
                        }
                    }
                }
            }
        }

        public static void ForEachNode(List<List<Node>> network, bool ignoreInputs, Action<Node> accessor)
        {
            for (var layerIdx = ignoreInputs ? 1 : 0; layerIdx < network.Count; layerIdx++)
            {
                var currentLayer = network[layerIdx];
                for (var i = 0; i < currentLayer.Count; i++)
                {
                    var node = currentLayer[i];
                    accessor(node);
                }
            }
        }

        public static Node GetOutputNode(List<List<Node>> network)
        {
            return network[network.Count - 1][0];
        }
    }
}
