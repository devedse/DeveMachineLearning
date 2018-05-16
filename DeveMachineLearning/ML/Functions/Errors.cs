using System;
using System.Collections.Generic;
using System.Text;

namespace DeveMachineLearning.ML.Functions
{
    public static class Errors
    {
        public static SQUARE SQUARE = new SQUARE();
    }

    public class SQUARE : IErrorFunction
    {
        public double Error(double output, double target) => 0.5 * Math.Pow(output - target, 2);
        public double Der(double output, double target) => output - target;
    }
}
