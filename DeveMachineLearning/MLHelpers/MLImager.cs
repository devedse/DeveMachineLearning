using DeveMachineLearning.ML;
using DeveMachineLearning.MLImplementers;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Drawing;
using SixLabors.ImageSharp.Processing.Drawing.Brushes;
using SixLabors.ImageSharp.Processing.Drawing.Pens;
using SixLabors.ImageSharp.Processing.Text;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace DeveMachineLearning.MLHelpers
{
    public static class MLImager
    {
        public static Image<Rgba32> GenerateImage(List<List<Node>> network, Dictionary<string, bool> state, int size)
        {
            var img2 = new Image<Rgba32>(size, size);
            for (int x = 0; x < size; x++)
            {
                for (int y = 0; y < size; y++)
                {
                    double outx = x / (double)size * 2.0 - 1.0;
                    double outy = y / (double)size * 2.0 - 1.0;

                    var input = MLHelper.ConstructInput(state, outx, outy);
                    var lastNode = NetworkBuilder.ForwardProp(network, input);

                    //byte r = (byte)(255 * (1 - lastNode));
                    //byte g = 0;
                    //byte b = 0;

                    //=MAX((1-((A2+1)/2) / 2 * 3) *255; 0)
                    byte r = (byte)Math.Max((1 - ((lastNode + 1) / 2.0) / 2.0 * 3.0) * 255.0, 0);
                    //=MAX((1-((A2+1)/2) / 2 * 6) *128; 0)
                    byte g = (byte)Math.Max((1 - ((lastNode + 1) / 2.0) / 2.0 * 6.0) * 128.0, 0);
                    //=MIN(MAX((((A2+1 - 0,6666)/2 / 2 *3) ) *255; 0); 255)
                    byte b = (byte)Math.Min(Math.Max((((lastNode + 1 - (2.0 / 3.0)) / 2.0 / 2.0 * 3.0)) * 255.0, 0), 255);

                    img2[x, y] = new Rgba32(r, g, b);

                    //var pntX = ((pnt.X + 1.0) / 2.0) * size;
                    //var pntY = ((pnt.Y + 1.0) / 2.0) * size;
                }
            }

            return img2;
        }

        public static void AddPointsToImage(Image<Rgba32> image, List<Puntje> lijstje)
        {
            foreach (var pnt in lijstje)
            {
                var pntX = ((pnt.X + 1.0) / 2.0) * image.Width;
                var pntY = ((pnt.Y + 1.0) / 2.0) * image.Height;

                int pntXInt = (int)pntX;
                int pntYInt = (int)pntY;


                Rgba32 color = Rgba32.Black;


                if (pnt.PuntType == PuntType.Oranje)
                {
                    color = new Rgba32(255, 128, 0);
                }
                else
                {
                    color = new Rgba32(0, 0, 255);
                }

                var pen = Pens.Solid(Rgba32.White, 2);
                var poly = new SixLabors.Shapes.EllipsePolygon((float)pntX, (float)pntY, 3);
                image.Mutate(t => t.Fill(color, poly));
                image.Mutate(t => t.Draw(pen, poly));


                image[pntXInt, pntYInt] = color;
            }

        }
    }
}
