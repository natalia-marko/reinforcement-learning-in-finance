# Archive Directory

This directory contains archived/backup files that are no longer in active use.

## Files

- `new.ipynb.backup_*` - Experimental/backup version of walk-forward validation notebook
  - Older architecture: Technical SAC + Sentiment placeholder
  - Self-contained with embedded environment classes
  - Superseded by `01_base_agents.ipynb` with modular architecture

- `super_agent_env.py.backup_*` - Original SuperAgentEnv class file
  - Class moved to `environments.py`
  - Training function moved to `training.py`
  - Superseded by modular structure

- `meta_agent_env.py.backup_*` - Original MetaAgentEnv class file
  - Class moved to `environments.py`
  - Training function moved to `training.py`
  - Superseded by modular structure

## Purpose

Files here are kept for reference but should not be used in production workflows.

## Current Structure

- `environments.py` - All environment classes (Base, Sentiment, Technical, Super, Meta)
- `training.py` - All training functions (train_super_agent_sac, train_meta_agent)

