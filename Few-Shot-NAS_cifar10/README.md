# Few-shot NAS on CIFAR10

## Environment Requirements
```
python >= 3.6, numpy >= 1.9.1, torch >= 1.5.0 
```

## Download models
Download the large pre-trained model(45.5M) checkpoint from <a href=https://drive.google.com/drive/u/0/folders/1KVGozie7jiqMp9kbaFIzQwb-sMGZqb2r>here</a>, and place it on folder large. 

Download the small pre-trained model(3.79M) checkpoint from <a href=https://drive.google.com/drive/u/0/folders/1kIl7bol9GoA-oeNywCjGgDSQ9AkOo19s>here</a>, and place it on folder small. 


## Test Models

- For small model
```
python test.py –arch small
```

- For large model
```
python test.py –arch large
```

The top 1 accuracy of small model is 98.29%. The top 1 accuracy of large model is 98.72%.  




























