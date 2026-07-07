"""
seanet package - my new code for the SEA-Net project.

The idea of the project: take the MILLET paper's code (the "millet" folder) and build a smaller
model on top of it that keeps similar accuracy and interpretability. All of my own code lives
here in "seanet". The "millet" folder stays as the original code from the paper, except for two
tiny fixes that I list in the README ("Adjustments to millet"). Keeping it this way means anyone
can compare my repo to the original MILLET repo and see exactly what I changed.

The four modules in this package (each one does one job):
  * seanet.data     - loads a dataset by name, fixes the ~15 broken UCR datasets, checks the
                      file looks right, and writes a small summary row per dataset.
  * seanet.model    - the SEA-Net network (a small encoder + MILLET's Additive pooling).
  * seanet.train    - the training loop, early stopping, and evaluation, per dataset.
  * seanet.results  - saves results, remembers what is already done, and compares to MILLET.

How they fit together (the flow):
  data (load a dataset) -> model (build the network) -> train (fit + score it)
  -> results (save the numbers + compare to MILLET).

You do not import these directly to run things. Everything is run from "main.py" in the repo
root, for example:  python main.py summary / params / webtraffic / single <name> / train / results
(run "python main.py -h" to see all the commands). The data analysis is in analysis.ipynb.
"""
