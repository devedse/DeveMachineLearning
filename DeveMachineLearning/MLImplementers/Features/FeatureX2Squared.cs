namespace DeveMachineLearning.MLImplementers.Features
{
    public class FeatureX2Squared : IFeature
    {
        public double Function(double x1, double x2)
        {
            return x2 * x2;
        }
    }
}
