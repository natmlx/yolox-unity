# YOLOX
[YOLOX](https://arxiv.org/abs/2107.08430) high performance general object detection.

## Installing YOLOX
Add the following items to your Unity project's `Packages/manifest.json`:
```json
{
  "scopedRegistries": [
    {
      "name": "NatML",
      "url": "https://registry.npmjs.com",
      "scopes": ["ai.natml", "ai.fxn"]
    }
  ],
  "dependencies": {
    "ai.natml.vision.yolox": "1.0.4"
  }
}
```

## Detecting Objects in an Image
First, create the YOLOX predictor:
```csharp
// Create the YOLOX predictor
var predictor = await YOLOXPredictor.Create();
```

Then detect objects in the image:
```csharp
// Given an image
Texture2D image = ...;
// Detect objects
YOLOXPredictor.Detection[] detections = predictor.Predict(image);
```

> The detection rects are provided in normalized coordinates in range `[0.0, 1.0]`. The score is also normalized in range `[0.0, 1.0]`.
___

## Requirements
- Unity 2021.2+

## Quick Tips
- Join the [NatML community on Discord](https://natml.ai/community).
- Discover more ML models on [NatML Hub](https://hub.natml.ai).
- See the [NatML documentation](https://docs.natml.ai/natml).
- Contact us at [hi@natml.ai](mailto:hi@natml.ai).

Thank you very much!