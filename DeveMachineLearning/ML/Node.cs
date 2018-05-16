using DeveMachineLearning.ML.Functions;
using System;
using System.Collections.Generic;
using System.Text;

namespace DeveMachineLearning.ML
{
    public class Node
    {
        public string Id { get; set; }
        public List<Link> InputLinks { get; } = new List<Link>();
        public double Bias { get; set; } = 0.1;
        public List<Link> Outputs { get; } = new List<Link>();
        public double TotalInput { get; set; }
        public double Output { get; set; }

        public double OutputDer { get; set; }
        public double InputDer { get; set; }

        public double AccInputDer { get; set; }
        public double NumAccumulatedDers { get; set; }

        public IActivationFunction ActivationFunction { get; set; }

        public Node(string id, IActivationFunction activationFunction, bool initZero)
        {
            Id = id;
            ActivationFunction = activationFunction;
            if (initZero)
            {
                Bias = 0;
            }
        }

        public double UpdateOutput()
        {
            this.TotalInput = this.Bias;
            for (int i = 0; i < this.InputLinks.Count; i++)
            {
                var link = this.InputLinks[i];
                this.TotalInput += link.Weight * link.Source.Output;
            }
            this.Output = this.ActivationFunction.Output(this.TotalInput);
            return this.Output;
        }
    }
}
