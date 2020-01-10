gather_stats.py:
------------
## Bayesian inference, part 1:
Gathering statistics from the xlsx files. This code reads a "Transmittance.xlsx" 
that contains entire training set and outputs a "trans.json" file that contains 
the statistics of mean average transmittance for each filter at each wavelength. 
In our case it contains 11 sheets, one sheet per nanomaterial filter (F1, F2, ..., F11). 
Each sheet contains transimttance spectrum that filter. The first column is wavelength, and from second column to the end the transmittance values are given. 

Furthermore, this file can read a second file "TestT.xlsx" containg the test samples.
The format of this file is the same as the Transmittance.xlsx file. Upon executing 
this code will read the test samples and put all of them in a "testT.csv" file in a 
single sheet in a way that each row is a sample of T vector including 11 transmittance
values t1, t2, ..., t11, one per filter. The total number of rows is equal to the total number of samples. If the test sample file is alredy at hand in the mentioned way and this part of the code is not needed the user can comment out the last two lines of code.




analysis.py:
----------------
## Bayesian inference, part 2:
This is the main code to perform the Bayesian inference. It reads the "trans.json" 
file as well as "testT.csv" file, and outputs the estimated wavelengths for 
real/synthesied test samples  using maximum a posteriori or MAP estimation. The estimated wavelengths for synthesized test samples can show how well the model is working on the training set itself. This way, user can calculate test error and training error. Each section of this code can accomplish a different task. First, the used needs to run the code, but it will not output any results. To obtain results the user needs to call specific funcitons defined in this code. The task of each function is explained right before fucntion's definition. First, run the code; then choose a function, and call it from console or comand line. The functions that output desired results are pointed out by "Call this function for:". 