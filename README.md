# Bathroom Stall Door Survey Analysis

This repository contains an end-to-end survey analysis pipeline for understanding user experiences with public bathroom stall doors. It loads raw responses, runs several text- and regex-based analyses, and produces summary statistics and visualizations.

## 📂 Repository Structure

project1Trends/
├── responses.csv # Raw survey data
├── bathroom_survey_analysis.png # Initial exploratory plot
├── enhanced_bathroom_survey_analysis.png # Final combined dashboard
├── README.md # This file
└── src/
├── main.py # Orchestrates all analyses & visualizations
└── test.py # (Optional) example usage or unit tests

## ⚙️ Requirements

- Python ≥ 3.8
- pandas
- numpy
- matplotlib
- seaborn

Install with:

```bash
pip install pandas numpy matplotlib seaborn


🚀 Quick Start
	1.	Clone the repo
        git clone https://github.com/RamonsArchive/TextAnalysisDSGN1.git
        cd TextAnalysisDSGN1

    2.	Install dependencies
        pip install -r requirements.txt

    3.	Run the analysis
        python src/main.py

This will:
	•	Load and clean responses.csv
	•	Print summary stats and key insights to the console
	•	Generate a set of visualizations and save them as enhanced_bathroom_survey_analysis.png

	4.	Inspect the output
	•	Check the console for detailed counts and categories
	•	View enhanced_bathroom_survey_analysis.png for the final dashboard

✨ Features
	•	Privacy & Gap Analysis
	•	Door Swing Preference Detection (inward vs. outward, sentiment-aware)
	•	Gender-Based Emotional Responses
	•	Lock Security Assessment
	•	Clean, Annotated Visualizations

🛠️ How It Works
	•	load_and_clean_data(): Reads responses.csv, strips column names, and prints a quick head().
	•	analyze_privacy_gaps(): Counts gap-related keywords and sentiment.
	•	analyze_door_swing_preferences(): Uses regex patterns to categorize preferences and sentiment.
	•	analyze_gender_differences(): Breaks down emotional language and avoidance behavior by gender.
	•	analyze_lock_security(): Tallies lock-related keywords and issues.
	•	create_visualizations(): Builds a 2×3 grid of plots summarizing key insights.

📝 Customization
	•	Add or refine regex patterns in analyze_swing_preference_optimized() for more precise categorization.
	•	Adjust keyword lists in each analysis function to tailor to your survey’s phrasing.
	•	Modify create_visualizations() to change chart types, colors, or layout.
```
