# COVYT: Coronavirus YouTube Speech Dataset Baseline Code

This codebase contains code for the baseline experiments included in the following publication:

```
@article{Triantafyllopoulos22-COVYT,
  title={COVYT: Introducing the Coronavirus YouTube and TikTok speech dataset featuring the same speakers with and without infection},
  author={Triantafyllopoulos, Andreas and Semertzidou, Anastasia and Song, Meishu and Pokorny, Florian B and Schuller, Bj{\"o}rn W},
  journal={arXiv preprint arXiv:2206.11045},
  year={2022}
}
```

which introduced the COVYT dataset:

```
@dataset{andreas_triantafyllopoulos_2022_6962930,
  author       = {Andreas Triantafyllopoulos and
                  Anastasia Semertzidou and
                  Meishu Song and
                  Florian B. Pokorny and
                  Björn W. Schuller},
  title        = {{COVYT: Introducing the Coronavirus YouTube speech 
                   dataset featuring the same speakers with and
                   without infection}},
  month        = sep,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.6962930},
  url          = {https://doi.org/10.5281/zenodo.6962930}
}
```

To run the code, first install the requirements via:

```bash
pip install -r requirements.txt
```

Then download and extract the COVYT dataset.
When running `run.sh`, you need to specify the downloaded data and an output directory for your experiments.