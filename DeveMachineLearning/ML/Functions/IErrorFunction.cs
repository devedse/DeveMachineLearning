using System;
using System.Collections.Generic;
using System.Text;

namespace DeveMachineLearning.ML.Functions
{
    public interface IErrorFunction
    {
        double Error(double output, double target);
        double Der(double output, double target);
    }
}
