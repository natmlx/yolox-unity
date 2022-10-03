/* 
*   YOLOX
*   Copyright (c) 2022 NatML Inc. All Rights Reserved.
*/

namespace NatML.Examples {

    using UnityEngine;
    using NatML.Devices;
    using NatML.Devices.Outputs;
    using NatML.Features;
    using NatML.Vision;
    using NatML.Visualizers;
    using Stopwatch = System.Diagnostics.Stopwatch;

    public sealed class YOLOXSample : MonoBehaviour {

        [Header(@"UI")]
        public YOLOXVisualizer visualizer;

        private CameraDevice cameraDevice;
        private TextureOutput cameraTextureOutput;

        private MLModelData modelData;
        private MLModel model;
        private YOLOXPredictor predictor;

        async void Start () {
            // Request camera permissions
            var permissionStatus = await MediaDeviceQuery.RequestPermissions<CameraDevice>();
            if (permissionStatus != PermissionStatus.Authorized) {
                Debug.Log(@"User did not grant camera permissions");
                return;
            }
            // Get a camera device
            var query = new MediaDeviceQuery(MediaDeviceCriteria.CameraDevice);
            cameraDevice = query.current as CameraDevice;
            // Start the camera preview
            cameraTextureOutput = new TextureOutput();
            cameraDevice.StartRunning(cameraTextureOutput);
            // Display the camera preview
            var cameraTexture = await cameraTextureOutput;
            visualizer.image = cameraTexture;
            // Create the YOLOX predictor
            modelData = await MLModelData.FromHub("@natsuite/yolox");
            model = modelData.Deserialize();
            predictor = new YOLOXPredictor(model, modelData.labels);
        }

        void Update () {
            // Check that predictor has been created
            if (predictor == null)
                return;
            // Create image feature
            var imageFeature = new MLImageFeature(cameraTextureOutput.texture);
            (imageFeature.mean, imageFeature.std) = modelData.normalization;
            imageFeature.aspectMode = modelData.aspectMode;
            // Detect
            var detections = predictor.Predict(imageFeature);
            // Visualize
            visualizer.Render(detections);
        }

        void OnDisable () {
            // Dispose the model
            model?.Dispose();
        }
    }
}