Abstract Scoring Toolkit
========================

This repository contains a small toolkit for prioritising scientific abstracts.
It converts CSV files of abstracts into graded and sorted outputs using a
weighted keyword dictionary and semantic similarity.

Requirements
------------
Install the required dependencies:

```
pip install -r requirements.txt
python -m spacy download en_core_sci_sm
```

Configuration
-------------
The scoring behaviour is controlled by a JSON configuration file. A minimal
example is provided in ``config.json``. Adjust the keyword weights and other
parameters to match your project.

Directory Layout
----------------
The default workflow expects an ``Articles`` directory with subfolders for each
database (e.g. ``PM`` or ``EMBASE``):

```
Articles/<DATABASE>/Scored_articles/
Articles/<DATABASE>/Graded_articles/
Articles/<DATABASE>/Graded_sorted_articles/
```

These folders are created automatically when running the scripts, but the
``Scored_articles`` folder must contain the input CSV file.

Usage
-----
Grade and sort a CSV of abstracts:

```
python mainGrader.py --current-iteration 1 --database PM \
    --base-dir Articles --config config.json --delimiter ','
```

This runs ``CSVgrader`` to score each abstract and ``CSVsortbyscore`` to produce
an additional file sorted by score.  The paths and parameters are fully
customisable through command-line options.

Advanced Parameter Tuning
-------------------------
The ``GradingTweaking`` module performs Monte Carlo optimisation of the keyword
weights against a reference set of included/excluded abstracts:

```
python GradingTweaking.py --input path/to/scored.csv --output tuned.csv \
    --config config.json --delimiter ','
```

The updated configuration is written back to the supplied config file.

