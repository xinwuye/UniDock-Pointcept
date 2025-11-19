- successful reproduction:
    ```bash
    conda env create -f environment.yml --verbose
    conda activate pointcept-torch2.5.0-cu12.4
    git@github.com:Dao-AILab/flash-attention.git
    cd flash-attention
    MAX_JOBS=4 FLASH_ATTN_CUDA_ARCHS="80;90" python setup.py install
    cd ..
    pip install -v --no-build-isolation ./libs/pointops
    pip install -v --no-build-isolation ./libs/pointgroup_ops
    ```
- to download the checkpoint and data:
    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    huggingface-cli download Pointcept/PointTransformerV3 scannet-semseg-pt-v3m1-0-base/model/model_best.pth --local-dir exp/scannet --local-dir-use-symlinks False

    # data
    huggingface-cli login
    huggingface-cli download Pointcept/scannet-compressed --repo-type dataset --local-dir /public/home/group_gaosh/xinwuye/UniDock/UniDock-Pointcept --local-dir-use-symlinks False
    mkdir -p data/scannet_processed
    PROCESSED_SCANNET_DIR=data/scannet_processed
    ln -s ${PROCESSED_SCANNET_DIR} data/scannet
    ```
- to do inference:
```bash
cd /public/home/group_gaosh/xinwuye/UniDock/UniDock-Pointcept
sh scripts/test.sh -g 4 \
  -d scannet \
  -c semseg-pt-v3m1-0-base \
  -n scannet-semseg-pt-v3m1-0-base \
  -w model_best
```