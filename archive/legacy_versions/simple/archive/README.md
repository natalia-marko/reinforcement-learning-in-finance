# RL in Finance - Simple Implementation

This project implements a Reinforcement Learning based portfolio management system.

## Structure

- `core/`: Core RL modules, data engineering, and training logic.
- `data/`: Directory for storing downloaded financial data.
- `models/`: Directory for saving trained RL models.
- `app.py`: FastAPI application for serving the model.
- `static/`: Static files for the web interface.
- Jupyter notebooks for experimentation and pipelines.
## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the API:
   ```bash
   uvicorn app:app --reload
   ```

3. Access the web interface at `http://localhost:8000/static/index.html` (if available) or check API docs at `http://localhost:8000/docs`.
