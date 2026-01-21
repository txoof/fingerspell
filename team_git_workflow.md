# Team Git Workflow Guide

## Purpose

This document explains how we work together safely in a shared GitHub repository.  
The goal is to avoid broken branches, lost work, and merge conflicts, while keeping the workflow simple and repeatable.

---

## Branch overview

We use three types of branches.

### main  
Client ready branch  
Nothing is pushed here directly  
All changes arrive here via approved pull requests

### development  
Shared working branch  
All feature work is based on this branch  
This branch is protected and requires a pull request

### feature branches  
Personal work branches  
Created from development  
Each branch contains one focused change

---

## Branch naming

Use lowercase and underscores only.

feature_short_description  
fix_short_description  
docs_short_description  

Examples  
feature_login_form  
fix_missing_null_check  
docs_readme_update  

Rules  
* One feature per branch  
* Do not reuse old branches  
* Do not work directly on development or main  

---

## The one approved workflow

This is the only supported workflow.

### Step 1. Update development

Before starting work, make sure development is up to date.

GitHub Desktop  
Fetch origin  
Switch to development  
Pull

---

### Step 2. Create a feature branch

Create a new branch from development.

GitHub Desktop  
Branch menu  
New branch  
Base branch: development  

---

### Step 3. Work on your feature

Make small, focused changes.  
Commit often with clear messages.

Rules  
* Do not edit files unrelated to your feature  
* Do not mix multiple features in one branch  

---

### Step 4. Stay in sync with development

At least once per day, update your branch with development.

GitHub Desktop  
Switch to your feature branch  
Fetch origin  
Branch menu  
Update from development  

This reduces merge conflicts later.

---

### Step 5. Open a pull request

When your work is ready:

* Push your branch  
* Open a pull request into development  
* Request one reviewer  
* Wait for approval  

All merges are squash merges.

---

### Step 6. Clean up

After your pull request is merged:

* Delete your feature branch  
* Switch back to development  
* Pull latest changes  

---

## Rules we always follow

* Never force push  
* Never rebase  
* Never work directly on development or main  
* Never fix merge conflicts in a hurry  
* Ask for help early if something looks wrong  

---

## Jupyter notebook rules

Notebooks are for:
* Experiments  
* Testing ideas  
* Exploration  

Production code belongs in:
* .py files or equivalent

Notebook rules:
* Only one person edits a given notebook at a time  
* Always pull development before opening a notebook  
* If a notebook conflict happens, stop and ask for help  
* Do not try to manually fix notebook conflicts unless you are confident  

Why this matters:  
Notebooks store outputs and metadata that cause frequent conflicts even when code looks unchanged.

---

## Troubleshooting common problems

### Problem: Your branch is behind development  
Solution:
* Switch to your feature branch  
* Update from development  
* Resolve conflicts if needed  
* Commit and push  

---

### Problem: Merge conflict appears  
Solution:
* Stop  
* Read the conflict message  
* Resolve one file at a time  
* Ask for help if unsure  

---

### Problem: You committed to the wrong branch  
Solution:
* Do not push  
* Ask for help  
This is easy to fix if caught early.

---

### Problem: Things feel very broken  
Solution:
* Stop  
* Do not force push  
* Do not delete anything  
* Ask for help  

---

# Mini Curriculum: Git Collaboration Session

## Audience

People comfortable using GitHub Desktop on solo projects but new to collaboration.

---

## Learning goals

* Understand which branch to work on  
* Create and update feature branches safely  
* Open pull requests correctly  
* Stay calm when conflicts happen  

---

## Part 1: Mental model (5 to 7 minutes)

Topics:
* What branches represent in this project  
* Why we never work directly on development  
* What a pull request actually does  

Teaching focus:
* Safety over speed  
* Process over cleverness  
* Mistakes are normal and recoverable  

---

## Part 2: Live walkthrough (10 to 15 minutes)

Instructor demonstrates:
* Updating development  
* Creating a feature branch  
* Making a small change  
* Updating from development  
* Opening a pull request  
* Squash merging  

Students observe only.  
Questions encouraged.

---

## Part 3: Hands on exercise (10 to 15 minutes)

Setup:
Everyone creates their own feature branch from development.

Task examples:
* Edit README  
* Add a comment to a file  
* Create a small text file  

Steps students perform:
* Create feature branch  
* Commit change  
* Update from development  
* Open pull request into development  

Instructor role:
* Walk around  
* Help with mistakes  
* Reinforce correct habits  

---

## Part 4: Conflict awareness demo (5 minutes, optional)

Instructor shows:
* A real merge conflict  
* How to slow down  
* How to resolve or ask for help  

Key message:  
Conflicts are normal. Panic causes damage.

---

## Optional follow up practice

* Two people edit the same file intentionally  
* Practice daily development updates  
* Group review of a pull request  
