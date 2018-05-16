using DeveMachineLearning.ML;
using DeveMachineLearning.ML.Functions;
using DeveMachineLearning.MLImplementers;
using DeveMachineLearning.MLImplementers.Features;
using System.Collections.Generic;

namespace DeveMachineLearning.MLHelpers
{
    public static class MLHelper
    {
        public static List<double> ConstructInput(Dictionary<string, bool> state, double x, double y)
        {
            var input = new List<double>();
            foreach (var inputName in INPUTS.INPUTSDICT)
            {
                if (state[inputName.Key])
                {
                    input.Add(INPUTS.INPUTSDICT[inputName.Key].Function(x, y));
                }
            }
            return input;
        }

        public static double GetLoss(List<List<Node>> network, Dictionary<string, bool> state, List<Puntje> dataPoints)
        {
            double loss = 0;
            for (var i = 0; i < dataPoints.Count; i++)
            {
                var dataPoint = dataPoints[i];
                var input = MLHelper.ConstructInput(state, dataPoint.X, dataPoint.Y);
                var output = NetworkBuilder.ForwardProp(network, input);
                loss += Errors.SQUARE.Error(output, dataPoint.Label);
            }
            return loss / dataPoints.Count;
        }
    }
}
