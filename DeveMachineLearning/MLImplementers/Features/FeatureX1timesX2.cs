namespace DeveMachineLearning.MLImplementers.Features
{
    public class FeatureX1timesX2 : IFeature
    {
        public double Function(double x1, double x2)
        {
            return x1 * x2;
        }
    }
}
