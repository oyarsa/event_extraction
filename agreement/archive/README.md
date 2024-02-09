# Agreement and other metrics calculation

The files in this directory used to calcualte the agreement between the human evaluation
and various evaluation metrics. They also contain the code for some correlation
statistics like Cohen, Spearman, etc.

These are not used anymore because they don't work correctly when there are entries with
duplicate context and annotation. This is mostly fine for FCR evaluation, but not for
TellMeWhy, especially the KnowWhy cache, since it will have duplicate in that sense.

As such, the code in this directory is not used anymore, but it is kept for reference.
