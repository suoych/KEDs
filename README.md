# KEDs

Implementation of the paper Knowledge-Enhanced Dual-stream Zero-shot Composed Image Retrieval (CVPR 2024)


### Preparation
1. Download the CC3M dataset (we transform the image format into image_byte format, you can use the raw image data as well).
2. Install the GPU version Faiss library, then random sample 0.5M image-text pairs from CC3M as Bi-modality knowledge. You can encode the database using CLIP model first and save them into a .pt file (refer to the code in src/eval_retrieval.py)
3. Install python environment
```
pip install -r requirements.txt
``` 
For other preparation, please refer to Pic2word project.

### Pretrained models and Random sampled databases

Please refer to [the huggingface repo](https://huggingface.co/LionheartzzZ/KEDs/tree/main), where the *cc_image_databases.pt* and *cc_text_databases.pt* contains the bi-modality knowledge features encoded by CLIP-VIT-L-14 and *image_stream.pt* and *text_stream.pt* are the example checkpoints for the two stream networks.

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
