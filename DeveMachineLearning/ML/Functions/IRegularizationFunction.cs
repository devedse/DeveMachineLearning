using System;
using System.Collections.Generic;
using System.Text;

namespace DeveMachineLearning.ML.Functions
{
    public interface IRegularizationFunction
    {
        double Output(double weight);
        double Der(double weight);
    }
}
