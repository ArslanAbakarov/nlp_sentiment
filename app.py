from shiny import App, ui, reactive, render,  Session

import os
import pandas as pd
import numpy as np
import pickle  # For loading the model
import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
# import torch

from sklearn.preprocessing import StandardScaler

# file = 'models/nlp-model-k'

prediction_model_path = 'models/a-dtmodel.pkl'
nystroem_path = 'models/a-nystroem.pkl'
vectorizer_path = 'models/a-vectorizer.pkl'

# from pycaret.clustering import predict_model

# Load the model using PyCaret's load_model
# kmeans = load_model('models/kmeans_model')
# kmeans = load_model('models/kmeans_model_customers_model')


# Define the static directory path to store the output CSV file
static_dir = os.path.join(os.path.dirname(__file__), "www")
os.makedirs(static_dir, exist_ok=True)  # Ensure the directory exists

# global variable df
df = pd.DataFrame()

# with open(file, 'rb') as file:
#     model = pickle.load(file)
    
# model = joblib.load('models/nlp-model-k')

model = pickle.load(open(prediction_model_path, 'rb'))
nystroem = pickle.load(open(nystroem_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))


# Define the UI
app_ui = ui.page_fluid(
    
    ui.HTML("""
        <link rel="stylesheet" href="https://fonts.cdnfonts.com/css/amazon-ember">
    """),
    
    ui.tags.style("""
        .introduction {
            margin-top: 20px;
        }
        
        table {
             border-style: none;
        }
        
        table .dataframe {
            border-collapse: collapse;
           
        }
        
        th {
            text-align: left;
            padding: 10px;
            text-transform: capitalize;
            border-bottom: 1px solid #ddd;
            
            color: #676363;
            
        }
        
        h1 {
            margin-bottom: 20px;
        }
        
        p {
            margin-bottom: 0;
        }
        
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
            font-weight:600;
        }
        
        .parameters-container {
            margin-top: 20px;
            border-radius: 10px;
            border-color: #ddd;
            border-style: solid;
            border-width: 2px;
            padding: 27px;
        }
        
        .btn-default {
            background: black;
            color: white;
            border-radius: 10px;
        }
        
        .page-container {
            padding-bottom: 50px;
        }
        
        #predict {
            margin-bottom: 20px;
        }
        
        #download_button {
            margin-top: 15px;
            display: block;
        }
        
        #download_link {
            margin-top: 20px;
            margin-bottom: 10px;
            display: block;
        }
        
        .progress-bar {
            border-radius: 10px;
        }
        
        #file_progress {
            margin-top: 2px;
        }
        
        .form-group {
            margin-top: 20px;
        }
        
        .control-label {
            font-weight: bold;
        }
        
        h2 {
            font-size: 27px !important;
            font-weight: 600;
            margin-bottom: 40px;
        }
    """),
    
      
    ui.tags.head(
        # Link to a Bootswatch theme (Cerulean)
        # ui.tags.link(rel="stylesheet", href="css/bootstrap.css")
        
        # ui.tags.link(
        #     rel="stylesheet",
        #     href="css/custom.css"
        # ),
    ),
    
    # https://shiny.posit.co/py/api/core/ui.input_slider.html
    
    ui.div(
        
        ui.div(
          ui.div(
     
         ),
          class_="upper-bar"  
        ),
        
        # ui.div(
        #     # ui.output_image("logoimage"),
        #     class_="container",
        # ),
        
        # output hr
        
        # ui.hr(),
        
        ui.div(
                       
            class_="container introduction",
        ),
    
        ui.div(
            
            ui.div(
                 
                    class_="",
            ),
            
            ui.div (        
                ui.h2("Dataset for prediction"),
                # ui.h2("Please fill in the form:"),
                # Input fields for model features
                # ui.input_slider("bmi", "BMI", 0.0, 100.0, value=3.029167),
                ui.p("Model predicts text's sentiment. Please make sure to provide CSV file with 'review' column."),
                
                ui.HTML("Quick demo file can be found here <a href='sample.csv'>sample.csv</a>"),
                # ui for file upload, allow only csv
                ui.input_file("file", "Upload a CSV dataset", placeholder="No dataset selected", accept=[".csv"],  multiple=False),

                # Button to trigger prediction
                ui.input_action_button("predict", "Predict"),
              
                class_="parameters-container",
                
            ),
            
            ui.div (
                ui.h2("Predicted dataset"),
                
                ui.input_action_button("download_button", "Download CSV"),

                # Output the prediction
                # ui.output_text_verbatim("prediction"),
                ui.output_ui("download_link"), 
                
                ui.output_ui("prediction"),  
                class_="parameters-container",
            ),
            
            class_="container",
        ),
        class_="page-container",
    )
)

# Define the server logic
def server(input, output, session):
    
    # load dataset from csv
    @reactive.calc
    def parsed_file():
        file: list[FileInfo] | None = input.file()
        if file is None:
            return pd.DataFrame()
        return pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
            file[0]["datapath"]
        )

    @render.table
    def summary():
        df = parsed_file()
        print("rendered")
        
        
    # Define the prediction logic

    @render.text
    def prediction():
        
        df = parsed_file()        
        # Wait until the button is clicked
        if input.predict() == 0:
            return "Click 'Predict' to get the result."

        # predictions = model.predict(df['review'].tolist())[0]
        
        x_tdif = vectorizer.transform(df['review'])
        x_tdif_prep = nystroem.transform(x_tdif)
        predictions = model.predict(x_tdif_prep)
        
        # vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
        # predictions = model.predict(vectorizer.transform(df['review']))
        
        df['sentiment'] = predictions
        
        # replace 0 with negative and 1 with positive
        df['sentiment'] = df['sentiment'].replace({1: 'negative', 3: 'positive', 2: 'neutral', 0: 'irrelevant'})
        
        # df to html
        predictions_html = df.to_html()

        return ui.HTML(predictions_html);
    
   
    # Reactive variable to store download link visibility
    download_link_visible = reactive.Value(False)

    # Define download functionality
    @reactive.Effect
    @reactive.event(input.download_button)
    def save_csv_and_show_link():
        df = parsed_file()
        if df.empty:
            download_link_visible.set(False)
            return

        # Ensure static directory exists
        os.makedirs(static_dir, exist_ok=True)
        
        csv_path = "www/output.csv"
        
        # Save DataFrame to CSV
        df.to_csv(csv_path, index=False)
        
        # Set the visibility of download link
        download_link_visible.set(True)

    # Render the download link based on reactive visibility state
    @render.text
    def download_link():
        if download_link_visible():
            download_url = "output.csv"  # Link to the generated CSV
            return ui.HTML(f"<a href='{download_url}' download>Click here to download the CSV</a>")
        else:
            return "Click 'Download CSV' to generate the file."

    output.download_link = download_link
        
# Create the app object
app = App(app_ui, server, static_assets=os.path.join(os.path.dirname(__file__), "www") )