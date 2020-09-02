# Malware-Research
Malware research with machine learning under guidance of Professor Mark Stamp at SJSU. Results will be published in a paper and in this book on deep learning: http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid=95376&copyownerid=101185

Dataset: https://drive.google.com/drive/u/1/folders/1ltGZw3Rw0Z-w7MXE1ltArPmWsvfFJbjX

Processed Dataset: https://drive.google.com/drive/u/3/folders/1iWYumJtqTLFo2T9V0wLOvKgoBgmh64sn

Goal: Use ensemble learning and various models to classify malware into their respective families


Process:

- Extract all file names to classify and group them into their families
- Use Radare2 to disassemble each file and write the opcode sequence onto text files
- Create a large .csv file with all the opcode data
  - in the .csv file, we use the first 1000 opcodes as features for training
    -remove any malware samples that do not have 1k opcoes or are corrupted
- models:
  -classic:
    - random forest
    - adaboost
    - xgboost
    - svm
    - bagged svm
    - hmm
    - bagged hmm
    - boosted hmm
    - knn
    - mlp
    - voting
  -deep learning:
    - cnn
    - bagged cnn
    - boosted cnn
    - lstm
    - bagged lstm
    - boosted lstm
  -voting:
    - all bagged and boosted cnns
    - all bagged and boosted lstms
    - all bagged cnns and bagged lstms
    - all boosted cnns and boosted lstms
    - all bagged and boosted cnns and lstms
    - all deep learning and classic models combined
 Results:
 -https://drive.google.com/drive/u/1/folders/1vliGOjaUDsqGVy_sq191jorfYquIj7JP
    
