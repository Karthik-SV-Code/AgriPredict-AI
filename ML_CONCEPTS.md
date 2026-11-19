# ML Concepts in AgriPredict AI

## 1. Regression
Regression predicts a continuous numerical value. This project predicts crop yield rather than a class such as “high” or “low”.

## 2. Features and target
- **Features (X):** Area, crop type, year, rainfall, pesticide usage, and temperature.
- **Target (y):** `hg/ha_yield`.

The model learns a relationship between the input features and the known yield values in the training data.

## 3. One-Hot Encoding
Machine-learning models need numerical input. `Area` and `Item` are text categories, so One-Hot Encoding creates binary indicator columns for their categories. `handle_unknown="ignore"` allows prediction when a category was not present during training, although such predictions may be less reliable.

## 4. Random Forest Regression
A Random Forest trains multiple decision trees using different samples/features. Each tree predicts a yield, and the forest averages the tree predictions. Averaging helps reduce the instability of a single decision tree.

Important parameter used here:
- `n_estimators=100`: build 100 decision trees.
- `random_state=42`: make the train/test split and model randomness repeatable.

## 5. Train/test split
The dataset is split into:
- 80% training data — used to learn patterns.
- 20% testing data — held aside to evaluate predictions on rows not used for fitting.


## 6. Evaluation metrics

- **R² score:** compares the model with a baseline that predicts the mean target. Closer to 1 generally indicates a better fit; it can be negative if the model performs worse than that baseline.
- **MAE:** average absolute difference between actual and predicted yield. It is easy to interpret and uses the target's unit (`hg/ha`).
- **RMSE:** square root of mean squared error. It also uses `hg/ha` and gives larger errors more influence.

## 7. separate preprocessing and model steps
The Scikit-learn separate preprocessing and model steps keeps preprocessing and prediction together. The same One-Hot Encoding learned during training is automatically applied when the Flask app makes a prediction. This prevents the web app from having to recreate the feature conversion manually.

## 8. End-to-end flow

`CSV dataset → select X and y → encode categories → train/test split → Random Forest fit → predict test rows → calculate metrics → save pipeline → Flask loads pipeline → predict user input`

## 9. Limitations to mention

- Historical associations do not prove that a feature causes higher or lower yield.
- A random split can give optimistic results when similar locations/years appear in both sets.
- Predictions outside the dataset's ranges may be unreliable.
- The dataset does not include every factor affecting yield, such as soil properties, irrigation, seed variety, or extreme weather events.

## 10. What-if Scenario Analysis

The simulator changes rainfall, average temperature and pesticide usage while keeping crop, region and year unchanged. It sends both the original and modified inputs to the same trained model.

The difference between the predictions shows how the model output responds to changed inputs.
