# Talking_Data
This is the solution to TalkingData Mobile User Demographics kaggle competetion after the deadline.
Model:
  Split data into two parts,one with evets and the other one without events.
  For the data with events part,only nn is used(tow layers,512 and 128 hidden nodes).The final outcome is the result of linear optimization of using
  scipy minization method of 10 different datasets on the same nn model.
  As for the data without evets,three models are used(nn,logistic regression and lightgbm).
