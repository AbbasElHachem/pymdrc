# pymdrc
Multiplicative Discrete Random Cascade Model (MDRC)


First read rainfall data on a fine resolution (1hour) and aggregate the data
to higher time frequencies:
(input Level: 1day; upper Level: 12hrs; middle Level: 6hrs, lower level: 3hrs
lowest level: 1.5hrs).

Second disaggregating the data through a mutiplicative Discrete Random Cascade
model (MDRC) (1440min->720min->360min->180min->90min),
at the end, rainfall on a finer resolution will be simulated
using a higher time frequency data.

This MDRC is a Microcanonical model: it conserves volumes in every level.

The first step is to find the weights (W1 and W2) of every level, this is done
by finding if the volume (V0) in the upper level (60min) has fallen
in the first sub interval (V1=V0.W1) or the second (V2=V0.W2) or in both.

Finding model parameters:
For every recorded rainfall in the upper level if volume > threshhold (0.3mm)
find W1 = R1/R and (W2 = 1-W1).
A sample of W is obtained in every level, plot histogram to find distribution.

The weights represent a probability of how the rainfall volume is distributed,
Three possible values for the weights:
W1 = 0 means all rainfall fell in 2nd sub-interval P (W=0)
W1 = 1 means all rainfall fell in 1st sub-interval P (W=1)
0 < W1 < 1 means part of rainfall fell in 1st sub iterval and part in the 2nd
For calculating P01, the relation between the volumes and the weights is
modeled through a logistic regression.
For calculating the prob P (0<W<1) a beta distribution is assigned
and using the maximum likelihood method the parameter ß is estimated
for every cascade level and every station.
The MDRC baseline model has two parameters P01 and ß per level.

the MDRC unbounded model is introduced and allows relating the probability
P01 to the rainfall volume R through a logistic regression function
the parameters of the logisticRegression fct: a an b are estimated
using the maximum likelihood method. This is done by first identifying where
w is 0 or 1 and for these values, find the corresponding rainfall volume R
and use log(logisticRegression fct) and where w is between ]0:1[ use
log(1-logisticRegression fct), in that way the parameters a and b are estimated
using all of the observed weights. This is done for every station and every
cascade level, therefore the unbounded model has three parameters per level
a, b and beta. the value of beta is the same used in the baseline model


Analysing:
Once parameters are found, study the effect of the time and space on them:
First divide the events into 4 different boxes:
(Isolated: 010, Enclosed: 111, Followed: 011, Preceded: 110  ), plot them
Second extract the P01 for every month and plot it
