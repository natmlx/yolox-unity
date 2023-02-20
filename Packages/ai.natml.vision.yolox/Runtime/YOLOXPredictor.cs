/* 
*   YOLOX
*   Copyright © 2023 NatML Inc. All Rights Reserved.
*/

namespace NatML.Vision {

    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using UnityEngine;
    using NatML.Features;
    using NatML.Internal;
    using NatML.Types;

    /// <summary>
    /// YOLOX predictor for general object detection.
    /// This predictor accepts an image feature and produces an array of detections.
    /// </summary>
    public sealed class YOLOXPredictor : IMLPredictor<YOLOXPredictor.Detection[]> {

        #region --Types--
        /// <summary>
        /// Detection.
        /// </summary>
        public readonly struct Detection {

            /// <summary>
            /// Normalized detection rect.
            /// </summary>
            public readonly Rect rect;

            /// <summary>
            /// Detection label.
            /// </summary>
            public readonly string label;

            /// <summary>
            /// Normalized detection score.
            /// </summary>
            public readonly float score;

            public Detection (Rect rect, string label, float score) {
                this.rect = rect;
                this.label = label;
                this.score = score;
            }
        }
        #endregion


        #region --Client API--
        /// <summary>
        /// Detect objects in an image.
        /// </summary>
        /// <param name="inputs">Input image.</param>
        /// <returns>Detected objects.</returns>
        public unsafe Detection[] Predict (params MLFeature[] inputs) {
            // Check
            if (inputs.Length != 1)
                throw new ArgumentException(@"YOLOX predictor expects a single feature", nameof(inputs));
            // Check type
            if (!(inputs[0] is MLImageFeature imageFeature))
                throw new ArgumentException(@"YOLOX predictor expects an image feature", nameof(inputs));
            // Pre-process
            (imageFeature.mean, imageFeature.std) = model.normalization;
            imageFeature.aspectMode = model.aspectMode;
            // Predict
            var inputType = model.inputs[0] as MLImageType;
            using var inputFeature = (imageFeature as IMLEdgeFeature).Create(inputType);
            using var outputFeatures = model.Predict(inputFeature);
            // Marshal
            var logitsData = (float*)outputFeatures[0].data;      // (1,6300,85)
            var shape8 = new [] { inputType.height / 8, inputType.width / 8, 85 };
            var shape16 = new [] { inputType.height / 16, inputType.width / 16, 85 };
            var shape32 = new [] { inputType.height / 32, inputType.width / 32, 85 };
            var logits8 = new MLArrayFeature<float>(&logitsData[0], shape8);
            var logits16 = new MLArrayFeature<float>(&logitsData[logits8.elementCount], shape16);
            var logits32 = new MLArrayFeature<float>(&logitsData[logits8.elementCount + logits16.elementCount], shape32);
            var (widthInv, heightInv) = (1f / inputType.width, 1f / inputType.height);
            var candidateBoxes = new List<Rect>();
            var candidateScores = new List<float>();
            var candidateLabels = new List<string>();
            foreach (var (logits, stride) in new [] { (logits8, 8), (logits16, 16), (logits32, 32) })
                for (int j = 0, jlen = logits.shape[0], ilen = logits.shape[1]; j < jlen; ++j)
                    for (int i = 0; i < ilen; ++i) {
                        // Check
                        var score = logits[j,i,4];
                        if (score < minScore)
                            continue;
                        // Get class
                        var label = Enumerable
                            .Range(5, 80)
                            .Aggregate((p, q) => logits[j,i,p] > logits[j,i,q] ? p : q) - 5;
                        // Decode box
                        var cx = (i + logits[j,i,0]) * stride * widthInv;
                        var cy = 1f - (j + logits[j,i,1]) * stride * heightInv;
                        var w = Mathf.Exp(logits[j,i,2]) * stride * widthInv;
                        var h = Mathf.Exp(logits[j,i,3]) * stride * heightInv;
                        var rawBox = new Rect(cx - 0.5f * w, cy - 0.5f * h, w, h);
                        var box = imageFeature?.TransformRect(rawBox, inputType) ?? rawBox;
                        // Add
                        candidateBoxes.Add(box);
                        candidateScores.Add(score);
                        candidateLabels.Add(model.labels[label]);
                    }
            var keepIdx = MLImageFeature.NonMaxSuppression(candidateBoxes, candidateScores, maxIoU);
            var result = new List<Detection>();
            foreach (var idx in keepIdx) {
                var detection = new Detection(candidateBoxes[idx], candidateLabels[idx], candidateScores[idx]);
                result.Add(detection);
            }
            // Return
            return result.ToArray();
        }

        /// <summary>
        /// Dispose the predictor and release resources.
        /// </summary>
        public void Dispose () => model.Dispose();

        /// <summary>
        /// Create the YOLOX predictor.
        /// </summary>
        /// <param name="minScore">Minimum candidate score.</param>
        /// <param name="maxIoU">Maximum intersection-over-union score for overlap removal.</param>
        public static async Task<YOLOXPredictor> Create (
            float minScore = 0.4f,
            float maxIoU = 0.5f,
            MLEdgeModel.Configuration configuration = null,
            string accessKey = null
        ) {
            var model = await MLEdgeModel.Create("@natsuite/yolox", configuration, accessKey);
            var predictor = new YOLOXPredictor(model, minScore, maxIoU);
            return predictor;
        }
        #endregion


        #region --Operations--
        private readonly MLEdgeModel model;
        private readonly float minScore;
        private readonly float maxIoU;

        private YOLOXPredictor (MLEdgeModel model, float minScore, float maxIoU) {
            this.model = model;
            this.minScore = minScore;
            this.maxIoU = maxIoU;
        }
        #endregion
    }
}