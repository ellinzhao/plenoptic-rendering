# Local Setup

Follow the [PyTorch3D installation instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). 

In short, you need Anaconda (it is much harder to setup with pyenv) to create a virtual environment and install the needed packages:
```
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install -c pytorch pytorch=1.9.1 torchvision
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

Edits in the python files are followed by the comment `@ellin`
