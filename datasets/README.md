# Datasets

Use this directory for evaluation and failure-analysis assets.

Recommended layout:

```text
datasets/
  coco/
    val2017/
    annotations/
      person_keypoints_val2017.json
  failure_tags/
    scene_tags.csv
```

`scene_tags.csv` can be used for failure analysis with columns such as:

```csv
image_id,tag
397133,low_light
397133,occlusion
```
