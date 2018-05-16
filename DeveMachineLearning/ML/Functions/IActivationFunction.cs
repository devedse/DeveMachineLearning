using System;
using System.Collections.Generic;
using System.Text;

namespace DeveMachineLearning.ML.Functions
{
    public interface IActivationFunction
    {
        double Output(double input);
        double Der(double input);
    }
}
