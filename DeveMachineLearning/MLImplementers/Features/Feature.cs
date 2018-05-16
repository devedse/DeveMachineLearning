using System;
using System.Collections.Generic;
using System.Text;

namespace DeveMachineLearning.MLImplementers.Features
{
    public interface IFeature
    {
        double Function(double x1, double x2);
    }
}
