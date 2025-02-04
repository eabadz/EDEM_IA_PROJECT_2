import streamlit as st
import numpy as np
import pandas as pd
from multiapp import MultiApp
from pages import eda, introduction, prediction

st.title('🤖 Loan Approval Prediction')

side_bar=st.sidebar
side_bar.write('Sidebar')

app = MultiApp()

# Add all your application here

app.add_app("Home", introduction.app)
app.add_app("Data", eda.app)
app.add_app("Model", prediction.app)

# The main app
app.run()
