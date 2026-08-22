## Object Detection Algorithms

1. Sliding Window
2. R CNN
3. Fast R CNN
4. Faster R CNN
5. YOLO (you only look once)

## How YOLO works

- X_train -> image with bounding box
- y_train ->
  - n _ n _ vector of len 7
  - each vector has [pc, Bx, By, Bw, Bh, C1, C2]
  - C1, C2 are classes like (cat and dog)
  - Bx, By are center point
  - Bw, Bh is width and height through center to make bounding box
  - pc is whether some classes is there or not

- Example 1: (dog detection)
  - X_train -> Image of Dog
  - y_train -> [1, 10,10, 40,40, 0, 1]
- Example 2: (cat detection)
  - X_train -> Image of Dog
  - y_train -> [1, 10,10, 40,40, 0, 1]
- Example 3: (None Detection)
  - X_train -> Image of Nothing
  - y_train -> [0, 10,10, 40,40, 0, 0]

## STEPS:

1. yolo divides image into matrix example (4*4 or 19*19 or 90\*90)
2. Each grid will give len of 7 vector that represents what that grid represents
3. when we trained assume we divide into 4 _ 4 grid then output will be 4 _ 4 \* 7
4. when predicting the image then output also will be 4 _ 4 _ 7 which tells you that what is inside each grid of 4 \* 4 grid by getting len 7 vector

## what if multiple bounding boxex ?

- we used intersection of union (here we get operlapping area)
