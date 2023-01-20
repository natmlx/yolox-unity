/* 
*   YOLOX
*   Copyright © 2023 NatML Inc. All Rights Reserved.
*/

namespace NatML.Examples {

    using UnityEngine;
    using NatML.VideoKit;
    using NatML.Vision;
    using NatML.Visualizers;

    public sealed class YOLOXSample : MonoBehaviour {

        [Header(@"VideoKit")]
        public VideoKitCameraManager cameraManager;

        [Header(@"UI")]
        public YOLOXVisualizer visualizer;

        private MLModelData modelData;
        private MLModel model;
        private YOLOXPredictor predictor;
        
        async void Start () {
            // Fetch the YOLOX model data
            modelData = await MLModelData.FromHub("@natsuite/yolox");
            // Create the model
            model = new MLEdgeModel(modelData);
            // Create the YOLOX predictor
            predictor = new YOLOXPredictor(model, modelData.labels);
            // Listen for camera frames
            cameraManager.OnCameraFrame.AddListener(OnCameraFrame);
        }

        private void OnCameraFrame (CameraFrame frame) {
            // Create image feature
            var feature = frame.feature;
            (feature.mean, feature.std) = modelData.normalization;
            feature.aspectMode = modelData.aspectMode;
            // Detect
            var detections = predictor.Predict(feature);
            // Visualize
            visualizer.Render(detections);
        }

        void OnDisable () {
            // Stop listening for camera frames
            cameraManager.OnCameraFrame.RemoveListener(OnCameraFrame);
            // Dispose the model
            model?.Dispose();
        }
    }
}