# ALPaCA development/analysis code

All code used to develop ALPaCA and analyze outputs.

code/to_copy/ folder
1. Used to copy raw files from different directory.
   
code/ folder
1. Preprocessing is done using .R files 00 to 03.
2. Competitors (MIMoSA, APRL, and ACVS) are run using 04 and associated with lesion candidates using 05
3. 06 used to analyze competitors vs ALPaCA
4. 07 used to create lesion/subtype masks using ALPaCA predictions
5. 08 used to check which gold-standard coordinates are missed
6. 09 used to develop ALPaCA R package

prl_pytorch/ folder
1. ALPaCA is developed using .py files 01 to 07
2. 08 to export trained ALPaCA network to R ALPaCA package
3. Exploration of individual-task networks (instead of multi-label network) using 09 to 11
4. 31 to 37 represent previous version of ALPaCA network
5. All .ipynb files were used to interactively test code that eventually ended up in .py files mentioned above.
