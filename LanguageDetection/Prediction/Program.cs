using Microsoft.ML;
using Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Prediction
{
    class Program
    {
        static readonly string[] classNames = { "German", "English", "French", "Italien", "Romanian", "Spanish" };

        static readonly IEnumerable<ClassificationData> predictSentimentsData = new[]
        {
            new ClassificationData
            {
                Text = "Hi there, this is Dirk speaking."
            },
            new ClassificationData
            {
                Text = "Hallo, mein Name ist Dirk."
            },
            new ClassificationData
            {
                Text = "Hola, mi nombre es Dirk."
            },
            new ClassificationData
            {
                Text = "Ciao, mi chiamo Dirk."
            },
            new ClassificationData
            {
                Text = "Bună ziua, numele meu este Dirk."
            },
            new ClassificationData
            {
                Text = "Bonjour, je m'appelle Dirk."
            }
        };

        const string modelPath = @".\Learned\Model.zip";

        public static void Main(string[] args)
        {
            Task.Run(async () =>
            {
                var model = await PredictAsync(modelPath, classNames, predictSentimentsData);

                Console.WriteLine();
                Console.WriteLine("Please enter another string to classify or just <Enter> to exit the program.");

                var input = string.Empty;

                while (string.IsNullOrEmpty(input = Console.ReadLine()) == false)
                {
                    IEnumerable<ClassificationData> predictInputSentiments = new[]
                    {
                        new ClassificationData
                        {
                            Text = input
                        }
                    };
                  
                    model = await PredictAsync(modelPath, classNames, predictInputSentiments, model);
                }

                Console.WriteLine("Press any key to end program...");
                Console.ReadKey();

            }).GetAwaiter().GetResult();
        }

        /// <summary>
        /// Predicts the test data outcomes based on a model that can be
        /// loaded via path or be given via parameter to this method.
        /// 
        /// Creates test data.
        /// Predicts sentiment based on test data.
        /// Combines test data and predictions for reporting.
        /// Displays the predicted results.
        /// </summary>
        /// <param name="model"></param>
        internal static async Task<PredictionModel<ClassificationData, ClassPrediction>> PredictAsync(
            string modelPath,
            string[] classNames,
            IEnumerable<ClassificationData> predicts = null,
            PredictionModel<ClassificationData, ClassPrediction> model = null)
        {
            if (model == null)
            {
              model = await PredictionModel.ReadAsync<ClassificationData, ClassPrediction>(modelPath);
            }

            if (predicts == null) // do we have input to predict a result?
                return model;

            // Use the model to predict the positive or negative sentiment of the data.
            IEnumerable<ClassPrediction> predictions = model.Predict(predicts);

            Console.WriteLine();
            Console.WriteLine("Classification Predictions");
            Console.WriteLine("--------------------------");

            // Builds pairs of (sentiment, prediction)
            IEnumerable<(ClassificationData sentiment, ClassPrediction prediction)> sentimentsAndPredictions =
                predicts.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

            foreach (var item in sentimentsAndPredictions)
            {
                string textDisplay = item.sentiment.Text;

                if (textDisplay.Length > 80)
                    textDisplay = textDisplay.Substring(0, 75) + "...";

                string predictedClass = classNames[(uint)item.prediction.Class];

                Console.WriteLine("Prediction: {0}-{1} | Test: '{2}'",
                    item.prediction.Class, predictedClass, textDisplay);
            }
            Console.WriteLine();

            return model;
        }
    }
}
