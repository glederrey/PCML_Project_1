README for the project 1 in PCML

Where to place the data-sets:

You can simply put the two data-sets (train.csv and test.csv) in this folder. You don't have to put them at some specific place.

Mandatory functions:

You will find the implementations of the six mandatory algorithms in the file implementations.py. All the additional functions for the mandatory algorithms can be found in the file called helpers.py. In addition, you can find a file called test_implementations.py. This file just tests the six mandatory functions on the train data-set. (The results are the one showed in Table I of our report)

Best result on Kaggle:

You can either use the run.py or the Python Notebook run.ipynb. The only thing is that we experienced some problems with the the python script (run.py) in the function np.linalg.solve. Therefore, if you want to get the exact same predictions we had on Kaggle, we recommand you to use the Python Notebook.

If you still want to use the run.py, you can run it like this:
    - run.py -da     -> It will create all the data-sets we used. (See report)
    - run.py -cv     -> It will run the cross-validation. Be careful it can take some time.
    - run.py -da -cv -> It will run the cross-validation and the data splitting. 
    
If you don't use the "-da" the first time, it will crash. And if you don't use the "-cv", it will use the best lambda and best degree we found and that are given in the report.

The additional functions for the run.py are given in the file called helpers_run.py

