#!/bin/bash
cd shared/datasets/coco
curl http://images.cocodataset.org/zips/train2017.zip -o train2017.zip
curl http://images.cocodataset.org/zips/val2017.zip -o val2017.zip
curl http://images.cocodataset.org/zips/test2017.zip -o test2017.zip
curl http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o annotations_trainval2017.zip

unzip train2017.zip
unzip annotations_trainval2017.zip