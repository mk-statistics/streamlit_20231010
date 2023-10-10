from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import category_encoders as ce
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import streamlit as st


# Define the Streamlit app
def main():
    # Set the title of your app
    st.title("Streamlit XGBoost Trainer")
    upload_values = st.file_uploader(
        "Upload for training-data (without labels).", type=["csv"]
    )

    upload_labels = st.file_uploader(
        "Upload for training-data (labels only).", type=["csv"]
    )

    if (
        st.button("Train XGBoost!")
        and upload_values is not None
        and upload_labels is not None
    ):
        train_values = pd.read_csv(upload_values)
        train_labels = pd.read_csv(upload_labels)

        # Drop building-ids
        train_values.drop(columns="building_id", inplace=True)
        train_labels.drop(columns="building_id", inplace=True)

        # Reduce due to XGBoost
        train_labels = train_labels - 1

        # Do the Split
        X_train, X_test, y_train, y_test = train_test_split(
            train_values, train_labels, random_state=42, test_size=0.2
        )

        # Cols to encode
        columns_to_encode = [
            "geo_level_1_id",
            "geo_level_2_id",
            "geo_level_3_id",
            "ground_floor_type",
            "roof_type",
            "land_surface_condition",
            "foundation_type",
            "other_floor_type",
            "position",
            "plan_configuration",
            "legal_ownership_status",
        ]

        # Make pipeline
        pipeline = make_pipeline(
            ce.OrdinalEncoder(cols=columns_to_encode), XGBClassifier()
        )

        with st.spinner("Wait for it..."):
            # Fit
            pipeline.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred = pipeline.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

        st.success(f"Accuracy: {accuracy:.2f}")
        st.write(f"XGBoost params:")
        st.json(pipeline.named_steps["xgbclassifier"].get_params())


# Run the app
if __name__ == "__main__":
    main()
