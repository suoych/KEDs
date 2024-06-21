# KEDs

Implementation of the paper Knowledge-Enhanced Dual-stream Zero-shot Composed Image Retrieval (CVPR 2024)
This is a raw version, we will further refine it.


### Preparation
1. Download the CC3M dataset (we use image_byte format data).
2. Install the GPU version Faiss library, then random sample 0.5M image-text pairs from CC3M as Bi-modality knowledge. You can encode the database using CLIP model first and save them into a .pt file (refer to the code in src/eval_retrieval.py)
3. Install python environment
```
pip install -r requirements.txt
``` 
For other preparation, please refer to Pic2word project.

### Training command

#### For raw folders
```
python -u src/main.py --save-frequency 1 --train-data="./cc3m/image_byte_224" --dataset-type directory --warmup 10000 --batch-size=128  --lr=1e-4 --wd=0.1  --epochs=30 --workers=6 --openai-pretrained --model ViT-L/14  --dist-url tcp://127.0.0.1:6102 --seed 999
```

### Demo inference command 
```
python src/demo.py --openai-pretrained --resume ./pic2word_model.pt --retrieval-data imgnet --query_file "./data/test.jpg" --prompts "a cartoon of *" --demo-out ./demo_result --gpu 1 --model ViT-L/14
```

### Evaluation for metrics command
```
python src/eval_retrieval.py --openai-pretrained --resume ./pic2word_model.pt --eval-mode cirr --gpu 0 --model ViT-L/14 --distributed --dist-url tcp://127.0.0.1:6101
```