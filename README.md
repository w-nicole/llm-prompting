# llm-prompting

## Setup

Use Satori.

Following 11/17/23: https://mit-satori.github.io/satori-ai-frameworks.html
```
cd /nobackup/users/$(whoami)
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-ppc64le.sh
sh Anaconda3-2022.05-Linux-ppc64le.sh -f -p /nobackup/users/$(whoami)/anaconda3
source ~/.bashrc
```
end cite

11/17/23: https://mit-satori.github.io/satori-ai-frameworks.html
```
conda config --prepend channels \
https://opence.mit.edu
```
end cite

Then, 

```conda create -n llm-prompting; conda activate llm-prompting```

11/17/23: https://stackoverflow.com/questions/48128029/installing-specific-build-of-an-anaconda-package

```conda install pytorch=1.10.2=cuda11.2_py38_1```

end cite

```pip3 install transformers pandas matplotlib```

```chmod u+x download_datasets.sh```

```./download_datasets.sh```

```git clone https://github.com/Tiiiger/bert_score.git```

```touch bert_score/__init__.py```



- Added new line