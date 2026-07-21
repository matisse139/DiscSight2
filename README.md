# DiscSight

DiscSight is a Streamlit-based Ultimate Frisbee analytics dashboard. It includes form analysis, trajectory mapping, player comparison, and an AI coaching assistant.

## Files

- `app.py`: Main Streamlit app for dashboard visualization and analysis.
- `train_model.py`: Model training script for ultimate biomechanics classification.
- `app1.py`: Secondary or experimental app file.
- `requirements.txt`: Python package dependencies.
- `ultimate_biomechanics_data.csv`: Synthetic dataset used for training.

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

## Notes

- Add `GROQ_API_KEY` to `.streamlit/secrets.toml` or your environment for API access.
- `app.py` uses computer vision and model inference for pose analysis and coaching prompts.
