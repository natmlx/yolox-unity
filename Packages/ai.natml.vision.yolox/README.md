# YOLOX
[YOLOX](https://arxiv.org/abs/2107.08430) high performance general object detection. This package requires [NatML](https://github.com/natmlx/NatML).

## Detecting Objects in an Image
First, create the YOLOX predictor:
```csharp
// Fetch the model data from NatML Hub
var modelData = await MLModelData.FromHub("@natsuite/yolox");
// Deserialize the model
var model = modelData.Deserialize();
// Create the YOLOX predictor
var predictor = new YOLOXPredictor(model, modelData.labels);
```

Then detect objects in the image:
```csharp
// Create image feature
Texture2D image = ...;
var imageFeature = new MLImageFeature(image); // This also accepts a `Color32[]` or `byte[]`
(imageFeature.mean, imageFeature.std) = modelData.normalization;
imageFeature.aspectMode = modelData.aspectMode;
// Detect objects
(Rect rect, string label, float score)[] detections = predictor.Predict(imageFeature);
```

> The detection rects are provided in normalized coordinates in range `[0.0, 1.0]`. The score is also normalized in range `[0.0, 1.0]`.
___

## Requirements
- Unity 2020.3+
- [NatML 1.0.11+](https://github.com/natmlx/NatML)

## Quick Tips
- Discover more ML models on [NatML Hub](https://hub.natml.ai).
- See the [NatML documentation](https://docs.natml.ai).
- Join the [NatML community on Discord](https://discord.gg/y5vwgXkz2f).
- Discuss [NatML on Unity Forums](https://forum.unity.com/threads/open-beta-natml-machine-learning-runtime.1109339/).
- Contact us at [hi@natml.ai](mailto:hi@natml.ai).

Thank you very much!