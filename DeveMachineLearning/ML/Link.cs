using DeveMachineLearning.Helpers;
using DeveMachineLearning.ML.Functions;

namespace DeveMachineLearning.ML
{
    public class Link
    {
        public string Id { get; set; }
        public Node Source { get; set; }
        public Node Dest { get; set; }
        public double Weight { get; set; } = RandomProvider.random.NextDouble() - 0.5;
        public bool IsDead { get; set; } = false;

        public double ErrorDer { get; set; }
        public double AccErrorDer { get; set; }
        public double NumAccumulatedDers { get; set; }

        public IRegularizationFunction RegularizationFunction { get; set; }

        public Link(Node source, Node dest, IRegularizationFunction regularizationFunction, bool initZero)
        {
            this.Id = $"{source.Id}-{dest.Id}";
            this.Source = source;
            this.Dest = dest;
            this.RegularizationFunction = regularizationFunction;
            if (initZero)
            {
                this.Weight = 0;
            }
        }
    }
}
