#!/bin/zsh

eval "$(conda shell.bash hook)"
conda activate garage37

nbstripout larch/doc/example/200_exampville.ipynb --keep-output --keep-count
nbstripout larch/doc/example/201_exville_mode_choice.ipynb --keep-output --keep-count
nbstripout larch/doc/example/202_exville_mc_logsums.ipynb --keep-output --keep-count
nbstripout larch/doc/example/203_exville_dest.ipynb --keep-output --keep-count
