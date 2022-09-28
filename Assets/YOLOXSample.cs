/* 
*   YOLOX
*   Copyright (c) 2022 NatML Inc. All Rights Reserved.
*/

namespace NatML.Examples {

    using UnityEngine;
    using NatML.Features;
    using NatML.Vision;
    using NatML.Visualizers;
    using Stopwatch = System.Diagnostics.Stopwatch;

    public sealed class YOLOXSample : MonoBehaviour {

        [Header(@"Prediction")]
        public Texture2D image;
        public bool gpu;

        [Header(@"UI")]
        public YOLOXVisualizer visualizer;

        async void Start () {
            Debug.Log("Fetching model data from NatML...");
            // Fetch the model data from NatML
            var modelData = await MLModelData.FromHub("@natsuite/yolox");
            modelData.computeTarget = gpu ? MLModelData.ComputeTarget.All : MLModelData.ComputeTarget.CPUOnly;
            // Deserialize the model
            using var model = modelData.Deserialize();
            // Create the YOLOX predictor
            using var predictor = new YOLOXPredictor(model, modelData.labels);
            // Create input feature
            var inputFeature = new MLImageFeature(image);
            (inputFeature.mean, inputFeature.std) = modelData.normalization;
            inputFeature.aspectMode = modelData.aspectMode;
            // Detect
            var watch = Stopwatch.StartNew();
            var detections = predictor.Predict(inputFeature);
            watch.Stop();
            // Visualize
            Debug.Log($"Detected {detections.Length} objects after {watch.Elapsed.TotalMilliseconds}ms");
            visualizer.Render(image, detections);
        }
    }
}