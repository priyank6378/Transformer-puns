# Transformer-puns
A seq2seq model used to give answers to pun questions. 

## Usage
### Training
To train the model, checkout the `train.ipynb` notebook.

### Inference
To use the model, run the following command:
```
python3 bot.py
```
This will start a chatbot in the terminal.

## Dataset
The dataset used for training is the [Pun of the Day](https://www.kaggle.com/datasets/jiriroz/qa-jokes) dataset from Kaggle. It contains around 38000 puns and their answers.

## Model
The model used is a transformer model. The model is trained on the puns and their answers. The model is trained for 50 epochs with a batch size of 16. The model is trained on a GPU (Gtx 1650). It took around 3 hours to train the model.

## Results
Question: Did you hear about the oyster who went to the ball?  
Original answer: He pulled a muscle  
Model answer: he pulled a muscle  

Question: What is a pirate's worst nightmare?  
Original answer: A sunken chest with no booty  
Model answer: a sunken chest with no booty  

Question: Which came first, the Chicken or the Egg?  
Original answer: The Rooster   
Model answer: well the car is just a black person who doesnt care about  

Question: Where did Susie go when her town got bombed?  
Original answer: Everywhere  
Model answer: everywhere  

Question: Why did the football coach go to Aldi?  
Original answer: To get his quarter back!   
Model answer: to get his palm tree  
