# ATRWEvalScript
 GroundTruth & Eval Scripts for ATRW Dataset

```
ATRWEvalScript
 ├─annotations
 |   (ground-truth files)
 ├─atrwtool
 |   (evaluation scripts)
 ├─sampleinput
 |   (sample input files for testing scripts & refernce)
 ├─README.md
```

## Install

The scripts run under python3. 

Shell command bellow installs required libs.

`pip3 install -r ./atrwtool/requirments.txt`

## Usage

`python ./atrwtool/main.py plain ./metric_epoch200_vit/query_expansion.json`

Where `[task]` is one of `detect, pose, plain, wild` for corresponding track.

For detailed result file format description, please refer to **Format Description** at https://cvwc2019.github.io/challenge.html



