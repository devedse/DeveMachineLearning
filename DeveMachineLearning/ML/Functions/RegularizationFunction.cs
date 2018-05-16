using System;

namespace DeveMachineLearning.ML.Functions
{
    public static class RegularizationFunction
    {
        public static L1 L1 = new L1();
        public static L2 L2 = new L2();
    }

    public class L1 : IRegularizationFunction
    {
        public double Output(double weight) => Math.Abs(weight);
        public double Der(double weight) => weight < 0 ? -1 : (weight > 0 ? 1 : 0);
    }

    public class L2 : IRegularizationFunction
    {
        public double Output(double weight) => 0.5 * weight * weight;
        public double Der(double weight) => weight;
    }
}
