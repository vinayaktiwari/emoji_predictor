# Programming assignment - 1 : Emoji prediction sentiment analysis
Problem Statement : Identify a reasonable "chat" data set and build a simple Python application to process the data, generate an Emoji suggestion tool (using a simple sentiment analysis model of your choice) which classifies the text and returns an appropriate Emoji. The application should have a very simple interface for the user to write a message to it and receive an Emoji as output.
This assignment covers the implementation of a Emotion Classification based on the input text message by the user

## Project structure
- Data cleaning and LSTM model trained on text data with 5 emoji classes(Jupiter notebook)
- Flask application that bridges HTML/CSS code with saved LSTM model
- HTML file that creates an user inteface for web application
## Setup
To run this project in your system, follow the steps below
- `Step1`:`git clone`
`https://github.com/vinayaktiwari/emoji_predictor.git`
- `Step2`:`$ pip install -r requirements.txt`
- `Step3`:To run the application run the command `$ python3 app.py` and then click on the web link [like this one -  * Running on http://127.0.0.1:5000] that will redirect you to a web page to run the predictions on local host 

## User interface 
Below is an image of user interface running on local host that takes inputs as texts and predicts an emoji associated to the sentiment of that text
![Screenshot from 2022-08-19 21-31-25](https://user-images.githubusercontent.com/26620896/185660168-f8b9a825-6215-4967-850f-32eec4fe24b3.png)
