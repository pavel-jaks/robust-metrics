# robust-metrics

Robust machine learning using different metrics

## Overview

This repository is a research repo
created for the purposes of my completion of masters degree on Faculty of Nuclear Sciences and Physical Engineering
of Czech Technical University in Prague.
It contains the academic output in the form of theses and notes
as well as implementation of the mathematical ideas presented in sections mentioned above.
Code is written in python programming language heavily depending on pytorch library.
The actual theses are written in czech.

## Abstract of the diploma thesis

There exists a problem in the field of machine learning called adversarial examples.
This is a phenomenon where even a small change of the input to a machine learning model
causes a big difference in the model output, which is unwanted in most cases.
In this work we study the questions concerning visual similarity metrics
and that in the context of existence and crafting of adversarial examples in the problem of image classification.
Our goal is to enlighten the way how such a visual similarity metric affects
the crafting process of adversarial examples and their final look.

## Repository structure

This repository contains two major directories:

- [code](./code)
  - Contains the implemetation of visual similarity metrics in subdirectory [metrics](./code/metrics)
  - Contains the output of adversarial examples creation process
  - For more details, go to [README.md](./code/README.md) file of the directory
- [TeX](./TeX)
  - Contains the theses (research project thesis and diploma thesis)
  - In czech language

## Acknowledgements

Special thanks to my supervisor Mgr. Lukáš Adam, Ph.D. and consultants Mgr. Vojtěch Čermák & Ing. Pavel Strachota, Ph.D.

---

Created by Bc. Pavel Jakš, 2022
