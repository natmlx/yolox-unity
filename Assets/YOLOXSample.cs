/* 
*   YOLOX
*   Copyright © 2023 NatML Inc. All Rights Reserved.
*/

namespace NatML.Examples {

    using UnityEngine;
    using NatML.Vision;
    using NatML.Visualizers;
    using VideoKit;

    public sealed class YOLOXSample : MonoBehaviour {

        [Header(@"VideoKit")]
        public VideoKitCameraManager cameraManager;

        [Header(@"UI")]
        public YOLOXVisualizer visualizer;

        private YOLOXPredictor predictor;
        
        async void Start () {
            // Create the YOLOX predictor
            predictor = await YOLOXPredictor.Create();
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
            // Dispose the predictor
            predictor?.Dispose();
        }
    }
}