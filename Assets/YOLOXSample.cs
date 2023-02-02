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
        private MLEdgeModel model;
        private YOLOXPredictor predictor;
        
        async void Start () {
            // Create the YOLOX model
            model = await MLEdgeModel.Create("@natsuite/yolox");
            // Create the YOLOX predictor
            predictor = new YOLOXPredictor(model);
            // Listen for camera frames
            cameraManager.OnCameraFrame.AddListener(OnCameraFrame);
        }

        private void OnCameraFrame (CameraFrame frame) {
            // Detect
            var detections = predictor.Predict(frame);
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