namespace DeveMachineLearning.MLImplementers.Features
{
    public class FeatureX1Squared : IFeature
    {
        public double Function(double x1, double x2)
        {
            return x1 * x1;
        }
    }
}
