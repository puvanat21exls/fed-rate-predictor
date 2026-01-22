Fed Rate Decision Predictor (v6)
A machine learning framework for forecasting FOMC interest rate decisions.

🧩 The Vision
This project was developed as a standalone "Fed Watcher" tool. It uses a Random Forest architecture to ingest monthly macroeconomic indicators (CPI, PCE, Unemployment, GDP) and output the probability of a Hike, Hold, or Cut.

🧬 The "Bio-Quant" Logic
Applying the same rigorous cross-validation techniques used in genomic sequence analysis, I implemented a TimeSeriesSplit to ensure the model never "cheats" by seeing future data during the training phase. This mirrors the integrity required in biomolecular research at UCSC.

📊 Performance & Insights
Performance Facts & Technical Integrity
Historical Accuracy: Achieved a baseline of ~70% correct predictions on FOMC rate decisions during stable economic regimes.

Model Sensitivity: The Random Forest architecture excels at capturing "averages" but shows a documented "lag" during major macroeconomic shifts or sudden policy pivots.

Validation Method: Utilized TimeSeriesSplit for cross-validation, ensuring zero data leakage from future time steps into past training sets.

Feature Weights: The model places the highest importance on CPI (Inflation) and Unemployment data, mirroring the Federal Reserve's "Dual Mandate".

Probability-Based Output: Rather than a simple binary "Hike/Hold," the model outputs a probability distribution across Hike (1), Hold (0), and Cut (2) to assist in risk assessment.

CME Alignment: While the CME FedWatch Tool uses Fed Funds Futures pricing (market-based), this model is purely data-based, providing a "Fundamentalist" perspective on the economy.
