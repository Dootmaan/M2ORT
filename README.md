# M2ORT: Many-To-One Regression Transformer for Spatial Transcriptomics Prediction from Histopathology Images

This is a preview of the code for M2ORT. Please do not redistribute it without contacting the authors as we have not included the LICENSE file yet.

Please note that this is NOT the code for our AAAI 2025 work [M2OST](https://github.com/Dootmaan/M2OST). M2OST has a different network structure design, so the pretrained weights in this repo won't work on M2OST. We are still sweeping the code of M2OST right now, so please wait for that part of release with patient. You are welcome to send a pull request for any repos.

### 1. Clone the code

Clone the code by running:

```
git clone git@github.com/Dootmaan/M2ORT.git
```

then cd into this directory.

### 2. Prepare your datasets

Download the datasets from their official site.

* HBC: https://data.mendeley.com/datasets/29ntw7sh4r/5.
  * Make sure you have also downloaded [this file](https://www.genenames.org/cgi-bin/download/custom?col=gd_hgnc_id&col=gd_app_sym&col=gd_app_name&col=md_ensembl_id&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit) before using the HBC dataset.
* HER2+: https://zenodo.org/records/3957257#.Y4LB-rLMIfg.
  * The files are encrypted by 7z. To decrypt these files, use the following passwords:
    * count matrices and images: zNLXkYk3Q9znUseS
    * meta data and spot selection: yUx44SzG6NdB32gY
* cSCC: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144240

### 3. Train or validate the model

Start training the M2ORT model using the following command:

```
CUDA_VISIBLE_DEVICES=0,1 nohup python3 -u train_m2ort.py --dataset HBC --dataset_path='/your/path/to/dataset' --variant [small,base,large] --checkpoint './weights/your_checkpoint.pth' >train_m2ort.log 2>&1
```
