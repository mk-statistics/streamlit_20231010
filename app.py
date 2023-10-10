import streamlit as st
import pandas as pd
import plotly.express as px


# Define the Streamlit app
def main():
    # Set the title of your app
    st.title("Streamlit Demo App")
    st.text("This is a test!")
    st.sidebar.radio("Select one", [1, 2])
    st.sidebar.multiselect("buy", ["milk", "apples", "banana"])
    st.sidebar.time_input("Meeting time")

    # Add a text input field
    user_input = st.text_input("Enter some text:")

    # Add a button
    if st.button("Submit"):
        # Display the entered text below the button
        st.write(f"You entered: {user_input}")

    col1, col2 = st.columns(2)
    col1.write("Col1")
    col2.write("Col2")

    DATA_URL = "https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
    nrows = 100

    @st.cache_data
    def load_data(DATA_URL, nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        data.rename(columns={"Lat": "LAT", "Lon": "LON"}, inplace=True)
        return data

    data = load_data(DATA_URL, nrows)
    st.dataframe(data)
    st.map(data.loc[:, ["LAT", "LON"]])

    df = px.data.iris()
    fig = px.parallel_coordinates(
        df,
        color="species_id",
        labels={
            "species_id": "Species",
            "sepal_width": "Sepal Width",
            "sepal_length": "Sepal Length",
            "petal_width": "Petal Width",
            "petal_length": "Petal Length",
        },
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=2,
    )
    st.plotly_chart(fig)


# Run the app
if __name__ == "__main__":
    main()
