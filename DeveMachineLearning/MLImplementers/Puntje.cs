using System;
using System.Collections.Generic;
using System.Text;

namespace DeveMachineLearning.MLImplementers
{
    public class Puntje
    {
        public double Label
        {
            get
            {
                if (PuntType == PuntType.Oranje)
                {
                    return -1;
                }
                return 1;
            }
        }

        public PuntType PuntType { get; set; }

        public double X { get; set; }
        public double Y { get; set; }
    }
}
