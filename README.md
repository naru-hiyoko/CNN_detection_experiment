# CNN detection experiment
CNN の featuremaps から位置推定を行う. 


ワークスペースを用意 <br>
.
├── CNN_detection_experiment
│   ├── Myutils.py
│   ├── README.md
│   ├── annotation.py
│   ├── create_samples.py
│   ├── detecter.py
│   ├── test_net.py
│   └── train_net.py
└── data
    ├── Annotations
    │   └── e1b9cb06-e9e3-440a-8d90-eb952fe995e1.xml
    ├── Images
    │   └── e1b9cb06-e9e3-440a-8d90-eb952fe995e1.JPEG
    ├── pkl
    │   └── data_2.pkl
    └── snapshot
        └── trained_30.model

annotation.py <br>
前景物体にバウンディングボックスをつけるアノテーションに利用. (要 wxpython, opencv)

create_samples.py <br>
前景領域と背景からMSERを利用してパッチ(48, 48)を切り出す.

python train_net.py で学習

test_net.py <br>
scipy によるラベリングアルゴリズム適用まで. 
