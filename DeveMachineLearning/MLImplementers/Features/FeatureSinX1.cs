using System;

namespace DeveMachineLearning.MLImplementers.Features
{
    public class FeatureSinX1 : IFeature
    {
        public double Function(double x1, double x2)
        {
            return Math.Sin(x1);
        }
    }
}
