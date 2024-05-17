# FinCausal data

This is data from multiple years of the [FinCausal
task](https://github.com/yseop/YseopLab).

The raw data is in the `raw` directory. They're CSV semicolon-separated files. Not every
year has all the files. Only 2020 has all dev/test/train, while 2021 has only dev/train
and 2022 has test/train.

Across these files, the evaluation files are identical, while the others have unique
entries. I've merged the files into a single CSV file for each year, which are in the
root of the `raw` directory.

The `convert_fincausal.py` script was used to convert these files to the tagged format
expected by the models here. In the case of the `test` file, they only provide the input
text without reference answers, so they are `null` in the output here.
