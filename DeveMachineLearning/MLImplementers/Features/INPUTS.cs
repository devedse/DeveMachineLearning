using System.Collections.Generic;

namespace DeveMachineLearning.MLImplementers.Features
{
    public static class INPUTS
    {
        public static Dictionary<string, IFeature> INPUTSDICT = new Dictionary<string, IFeature>()
        {
            { "x", new FeatureX1()},
            { "y", new FeatureX2()},
            { "xSquared", new FeatureX1Squared()},
            { "ySquared", new FeatureX2Squared()},
            { "xTimesY", new FeatureX1timesX2()},
            { "sinX", new FeatureSinX1()},
            { "sinY", new FeatureSinX2()}
        };
    }
}
