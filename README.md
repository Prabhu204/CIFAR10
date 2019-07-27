# Building a CNN classifier on CIFAR10 dataset
- CIFAR10 dataset is composed of 10 different classes as trainset <B>50000</B> and testset with <B>10000</B> samples.
- Each class was assigned with a number as fallows:
    <p>0  plane</p>
    <p>1  car</p>
    <p>2  bird</p>
    <p>3  cat</p>
    <p>4  deer</p>
    <p>5  dog</p>
    <p>6  frog</p>
    <p>7  horse</p>
    <p>8  ship</p>
    <p>9  truck</p>
- The model was composed of 3 Conv+ 2FC
- The model performance was test for each epoch.
    |          | Trainset | Testset |
    |----------|----------|---------|
    | Loss     | 0.1851   | 2.3147  |
    | Accuracy | 0.9334   | 0.6431  |
