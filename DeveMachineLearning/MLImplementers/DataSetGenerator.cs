using DeveMachineLearning.Helpers;
using System;
using System.Collections.Generic;
using System.Text;

namespace DeveMachineLearning.MLImplementers
{
    public static class DataSetGenerator
    {
        public static List<Puntje> Generate()
        {
            var lijstje = new List<Puntje>();

            int count = 50;
            AddPuntjes(lijstje, PuntType.Blauw, count, -1, -1, 0, 0);
            AddPuntjes(lijstje, PuntType.Oranje, count, 0, -1, 1, 0);
            AddPuntjes(lijstje, PuntType.Oranje, count, -1, 0, 0, 1);
            AddPuntjes(lijstje, PuntType.Blauw, count, 0, 0, 1, 1);

            return lijstje;
        }

        private static void AddPuntjes(List<Puntje> lijstje, PuntType puntType, int count, double minX, double minY, double maxX, double maxY)
        {
            for (int i = 0; i < count; i++)
            {
                var p = new Puntje()
                {
                    PuntType = puntType,
                    X = RandomProvider.random.NextDouble() * (maxX - minX) + minX,
                    Y = RandomProvider.random.NextDouble() * (maxY - minY) + minY
                };
                lijstje.Add(p);
            }
        }
    }
}
