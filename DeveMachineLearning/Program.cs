using DeveMachineLearning.Helpers;
using DeveMachineLearning.ML;
using DeveMachineLearning.ML.Functions;
using DeveMachineLearning.MLHelpers;
using DeveMachineLearning.MLImplementers;
using DeveMachineLearning.MLImplementers.Features;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace DeveMachineLearning
{
    class Program
    {
        private Random random = new Random();

        private Dictionary<string, bool> state = new Dictionary<string, bool>();

        private readonly int noise = 0;
        private readonly int batchSize = 10;
        private readonly double learningrate = 0.03;
        private readonly double regularizationRate = 0;

        private double lossTrain;
        private double lossTest;

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            var p = new Program();

            p.Go();


            Console.WriteLine("Application done");
            Console.ReadLine();

        }


        public Program()
        {


        }



        private void Go()
        {
            var lijstje = DataSetGenerator.Generate();


            int size = 100;
            var img = new Image<Rgba32>(size, size);

            foreach (var pnt in lijstje)
            {
                var pntX = ((pnt.X + 1.0) / 2.0) * size;
                var pntY = ((pnt.Y + 1.0) / 2.0) * size;

                int pntXInt = (int)pntX;
                int pntYInt = (int)pntY;

                if (pnt.PuntType == PuntType.Oranje)
                {

                    img[pntXInt, pntYInt] = new Rgba32(255, 0, 0);
                }
                else
                {
                    img[pntXInt, pntYInt] = new Rgba32(0, 0, 255);
                }
            }

            using (var fs = new FileStream("output.png", FileMode.Create, FileAccess.Write, FileShare.Read))
            {
                img.SaveAsPng(fs);
            }



            lijstje.Shuffle();

            var testData = lijstje.Take(lijstje.Count / 2).ToList();
            var trainData = lijstje.Skip(lijstje.Count / 2).ToList();


            foreach (var inp in INPUTS.INPUTSDICT)
            {
                state[inp.Key] = false;
            }

            state["x"] = true;
            state["y"] = true;
            //state["xTimesY"] = true;

            //var networkShape = new List<int>() { 2, 1, 1 };
            var networkShape = new List<int>() { 2, 8, 8, 8, 8, 8, 8, 1 };
            var inputIds = new List<string>() { "x", "y" };

            var network = NetworkBuilder.BuildNetwork(networkShape, Activations.TANH, Activations.TANH, null, inputIds, false);



            for (int i = 0; i < 100000; i++)
            {
                foreach (var trainPoint in trainData)
                {
                    var input = MLHelper.ConstructInput(state, trainPoint.X, trainPoint.Y);
                    NetworkBuilder.ForwardProp(network, input);
                    NetworkBuilder.BackProp(network, trainPoint.Label, Errors.SQUARE);
                    if ((i + 1) % batchSize == 0)
                    {
                        NetworkBuilder.UpdateWeights(network, learningrate, regularizationRate);
                    }
                }
                lossTrain = MLHelper.GetLoss(network, state, trainData);
                lossTest = MLHelper.GetLoss(network, state, testData);



                Console.WriteLine($"{i}: LossTrain: {lossTrain} LossTest: {lossTest}");
                MLImager.SaveAsImage(network, state, size);





                //for (int y = 1; y < network.Count; y++)
                //{
                //    var layer = network[y];
                //    Console.WriteLine(y);
                //    foreach (var node in layer)
                //    {
                //        var sb = new StringBuilder();
                //        foreach (var link in node.InputLinks)
                //        {
                //            sb.Append($"Lnk: {link.Weight} ");
                //        }
                //        Console.WriteLine($"   {sb.ToString()}");
                //    }
                //    Console.WriteLine();
                //}

                //double cccc = 0;
                //foreach (var testPoint in testData)
                //{
                //    var input = ConstructInput(testPoint.X, testPoint.Y);
                //    var result = NetworkBuilder.ForwardProp(network, input);

                //    var res = Math.Abs(result - testPoint.Label);
                //    cccc += res;
                //}
                //Console.WriteLine($"Res: {cccc}");

            }
        }







       
    }
}

