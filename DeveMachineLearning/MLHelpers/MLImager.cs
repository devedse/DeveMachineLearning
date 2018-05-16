using DeveMachineLearning.ML;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace DeveMachineLearning.MLHelpers
{
    public static class MLImager
    {
        public static void SaveAsImage(List<List<Node>> network, Dictionary<string, bool> state, int size)
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
            try
            {
                using (var fs = new FileStream("output2.png", FileMode.Create, FileAccess.Write, FileShare.Read))
                {
                    img2.SaveAsPng(fs);
                }
            }
            catch (Exception ex)
            {

            }
        }
    }
}
