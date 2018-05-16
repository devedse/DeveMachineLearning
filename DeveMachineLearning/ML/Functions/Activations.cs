using System;

namespace DeveMachineLearning.ML.Functions
{
    public static class Activations
    {
        public static TANH TANH = new TANH();
        public static RELU RELU = new RELU();
        public static SIGMOID SIGMOID = new SIGMOID();
        public static LINEAR LINEAR = new LINEAR();
    }

    public class TANH : IActivationFunction
    {
        public double Output(double input) => Math.Tanh(input);

        public double Der(double input)
        {
            var output = Output(input);
            return 1 - output * output;
        }
    }

    public class RELU : IActivationFunction
    {
        public double Output(double input) => Math.Max(0, input);

        public double Der(double input) => input <= 0 ? 0 : 1;
    }

    public class SIGMOID : IActivationFunction
    {
        public double Output(double input) => 1 / (1 + Math.Exp(-input));

        public double Der(double input)
        {
            var output = Output(input);
            return output * (1 - output);
        }
    }

    public class LINEAR : IActivationFunction
    {
        public double Output(double input) => input;

        public double Der(double input) => 1;
    }
}
