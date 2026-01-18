# Spatial Reconstruction of NBA Games using SportsVU and Play by Play Data

PLEASE UNZIP THE JSON FILE BEFORE RUNNING LOCALLY

The intention of this project is to reconstruct basketball games using gps coordinate data mapped to the court with an extension to potentially generating a "threat" score using novel methodology inspired by tomography.

The main files of interest are NBA_analysis.ipynb, QUT_Bootstrap.ipynb and funcs.py. More files will be added as the project progresses.

## NBA_Analysis

Features data cleaning, feature engineering and plotting of the movement data using several techniques and using probability to naively indicate areas of interest on the court. This file features only one game as it is designed to be prep for more intensive methods used in later notebooks. Some of the visuals generated are below.

### Full Game Movement Heatmap

<img width="1363" height="729" alt="image" src="https://github.com/user-attachments/assets/8cc3bf07-2a0d-400d-ab0a-533d2d3eeb69" />

### Per Quarter Heatmaps of Scoring Plays (Inverted)

<img width="1552" height="1073" alt="image" src="https://github.com/user-attachments/assets/376471cc-6089-4b39-bce0-f18b3e42fc8f" />

*Inverted so that all ball movements move from left to right for better interpretability and modeling down the line

### Per Quarter Heatmaps of Non-Scoring Plays (Inverted)

<img width="1549" height="1063" alt="image" src="https://github.com/user-attachments/assets/e438de74-be28-4ea7-b98b-425af7226a7a" />

### Conditional Probability of Scoring Heatmaps (Inverted)

<img width="1551" height="1071" alt="image" src="https://github.com/user-attachments/assets/f3c1045a-e24f-4734-9b5d-8f0bb4f99b71" />

There are more plots in the file and much more detail compared to here as this is a quick recap of what was completed for cleaning and visualizations.

## QUT_Bootstrap

The first step of modeling. Since we are attempting to use a novel method to model scoring threat, we first need to see if there is a signal to capture using bootstrapping and the Quantile Universal Threshold. This part is fairly math heavy and needs a bit more work to make it easier to understand exactly what is going on. More visuals provide below with some descriptions.

### Full Game QUT Distribution and Pointwise Measurements (Bigger number = higher assessed threat to score)

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/e162fa9c-41ef-44d8-8a63-06c97249cd5d" />

<img width="1789" height="443" alt="image" src="https://github.com/user-attachments/assets/f8f3ee33-3ab0-46f6-8161-0379ddb6339e" />

### Quarter 1 Through Quarter 4 QUT and Pointwise Measurements

#### Quarter 1
<img width="1793" height="1145" alt="image" src="https://github.com/user-attachments/assets/c261b348-0334-43aa-b09d-05a76f181953" />

#### Quarter 2
<img width="1793" height="1087" alt="image" src="https://github.com/user-attachments/assets/4b96c79f-8848-4404-a44a-686cdce5349e" />

#### Quarter 3
<img width="1791" height="1088" alt="image" src="https://github.com/user-attachments/assets/3a8f476d-69ec-4b65-956e-e493ed67e31a" />

#### Quarter 4
<img width="1789" height="1077" alt="image" src="https://github.com/user-attachments/assets/5e448098-b0b8-4982-865f-23ea9c8ad775" />

