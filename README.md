# BirdCLEF + BEATs — Repository Guidelines

This repository contains the code, experiments, documentation, and results for our
BirdCLEF project using Microsoft's pretrained **BEATs** audio model. The goal is to
make it easy for everyone on the team to contribute, experiment, share work, and
understand what has already been done.

## Table of Contents

1. [Purpose](#1-purpose)
2. [Contributors](#2-contributors)
3. [Working with Branches](#3-working-with-branches)
4. [Main Branch](#4-main-branch)
5. [Commit Messages](#5-commit-messages)
6. [Data and Large Files](#6-data-and-large-files)
7. [Data Processing](#7-data-processing)
8. [Avoid Data Leakage](#8-avoid-data-leakage)
9. [Experiments](#9-experiments)
10. [Experiment Names](#10-experiment-names)
11. [Evaluation](#11-evaluation)
12. [Training and Evaluation Data](#12-training-and-evaluation-data)
13. [BEATs](#13-beats)
14. [Frozen and Fine-Tuned Models](#14-frozen-and-fine-tuned-models)
15. [Code Organization](#15-code-organization)
16. [Document Useful Decisions](#16-document-useful-decisions)
17. [Keep Previous Results](#17-keep-previous-results)
18. [Testing](#18-testing)
19. [Communication](#19-communication)
20. [General Principle](#20-general-principle)

## 1. Purpose

This repository contains the code, experiments, documentation, and results for our
BirdCLEF project using Microsoft's pretrained BEATs audio model.

The goal is to make it easy for everyone on the team to contribute, experiment, share
work, and understand what has already been done.

## 2. Contributors

- Shadrack
- Bruce
- Gideon
- Arthur

Everyone is encouraged to contribute ideas, code, experiments, improvements, and
documentation.

## 3. Working with Branches

Branches are recommended when working on a new feature, experiment, or significant
change.

```text
feature/audio-preprocessing
feature/beats-finetuning
experiment/attention-pooling
bugfix/dataloader
```

For smaller changes, the team can use whatever workflow is most convenient, as long as
changes are communicated clearly.

## 4. Main Branch

Try to keep `main` in a usable state.

Before making major changes to `main`, make sure your code is reasonably tested and
won't unnecessarily break the rest of the project.

Pull requests are encouraged for larger changes because they make it easier for
everyone to review and discuss the work.

## 5. Commit Messages

Commit messages should give a reasonable idea of what changed.

Good examples:

```text
Add BirdCLEF preprocessing
Implement BEATs baseline
Add augmentation pipeline
Fix validation split
Add macro F1 evaluation
```

There is no need to follow an overly strict commit-message format. The main goal is
clarity.

## 6. Data and Large Files

Avoid committing large datasets, raw BirdCLEF audio, model checkpoints, or other files
that make the repository unnecessarily large.

- Use appropriate local folders or Git LFS when needed.
- Document where required datasets and checkpoints can be obtained.

## 7. Data Processing

Keep the original dataset unchanged whenever possible.

If we transform the data, implement the transformation in code or document what was done
so that another team member can reproduce it. Examples include:

- Resampling
- Windowing
- Normalization
- Augmentation
- Feature extraction
- Dataset splitting

## 8. Avoid Data Leakage

Be careful when creating training, validation, and test sets.

When recordings are split into multiple windows, try to ensure that windows from the
same original recording do not unintentionally end up in both the training and the
validation/test sets.

This is especially important because it can make the model appear more accurate than it
actually is.

## 9. Experiments

Treat experiments as part of the research process.

When practical, record the important settings for an experiment, such as:

- Model / checkpoint
- Window length
- Batch size
- Learning rate
- Number of epochs
- Frozen / unfrozen layers
- Augmentations
- Dataset split
- Metrics

The purpose is not to create paperwork. It is simply to make it easier to understand why
one experiment performed differently from another.

## 10. Experiment Names

Use names that make experiments easy to identify.

```text
exp001_frozen_beats
exp002_finetune_beats
exp003_attention_pooling
exp004_finetune_augmentation
```

This is especially useful when comparing results later.

## 11. Evaluation

Try to use a consistent evaluation process when comparing models. Useful metrics may
include:

- Accuracy
- Precision
- Recall
- F1 / Macro F1
- mAP, where appropriate
- Confusion matrix
- Validation / test loss

The exact metrics can evolve as we better understand the BirdCLEF task.

## 12. Training and Evaluation Data

As a general rule:

| Split           | Augmentation        |
| --------------- | ------------------- |
| Training data   | Augmentation allowed |
| Validation data | Normally unchanged  |
| Test data       | Normally unchanged  |

The purpose is to evaluate the model on data that has not been artificially modified.

## 13. BEATs

Keep track of which BEATs checkpoint and configuration are being used.

When changing the checkpoint or an important BEATs configuration, make a note of it so
that experiments can still be compared correctly.

## 14. Frozen and Fine-Tuned Models

Clearly identify whether an experiment uses **frozen BEATs** or **fine-tuned BEATs**.

If only certain Transformer layers are being trained, document that when relevant. For
example:

| Component    | State     |
| ------------ | --------- |
| Early layers | Frozen    |
| Later layers | Trainable |
| Classifier   | Trainable |

## 15. Code Organization

Try to keep related functionality together. A possible structure is:

```text
src/
├── data/
├── models/
├── training/
├── evaluation/
└── utils/
```

This is a guideline rather than a strict requirement. The structure can change as the
project develops.

## 16. Document Useful Decisions

Important technical decisions should be recorded somewhere in the repository when
possible. Examples:

- Why was a particular window size chosen?
- Why was an augmentation added or removed?
- Why was a pooling method changed?
- Why did fine-tuning help or hurt?
- Why was a particular checkpoint selected?

This helps us remember the reasoning behind the project rather than only the final code.

## 17. Keep Previous Results

Don't overwrite experiment results unnecessarily.

Keeping previous results makes it easier to compare approaches and understand how the
model evolved. For example:

```text
results/
├── exp001_frozen_beats/
├── exp002_finetune_beats/
└── exp003_attention_pooling/
```

## 18. Testing

Before sharing a significant change, try to verify that the relevant part of the project
still works.

For model changes, a useful basic check is:

```text
Audio
  ↓
BEATs
  ↓
Classifier
  ↓
Prediction
```

We don't need an elaborate testing system for every small change, but important
functionality should be checked before it becomes part of the main workflow.

## 19. Communication

When a change could affect someone else's work, let the team know. Examples:

- Changed dataset format
- Changed BEATs output
- Changed preprocessing
- Changed model architecture
- Changed dependencies
- Changed experiment configuration

Issues, pull requests, comments, and team communication can all be used for this.

## 20. General Principle

The repository should be:

> **Flexible enough for experimentation, but organized enough that everyone can
> understand what is happening.**

We're building a research project, so it is okay to try things, change direction, and
reorganize the code as we learn. The main goals are:

- Experiment freely
- Communicate clearly
- Keep useful results
- Avoid accidental data leakage
- Make important work reproducible
