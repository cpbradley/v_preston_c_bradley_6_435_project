README for the final project for 6.435. Prepared by Chris Bradley and Victoria Preston

This repository contains the source code for recreating the results presented in the included report.
The project goal was to emulate the odometric-data version of *Inference in the Space of Topological Maps: An MCMC-based Approach
https://smartech.gatech.edu/bitstream/handle/1853/38451/Ranganathan04iros.pdf*. In ```gaptopo_inference.py``` we include a wrapper function of Probabilistic Topological Maps (PTM) presented in this work, as well as our own extension call ed covPTMs. 

The main sript is fully commented; details about implementation can be found there or in the included report. The IPython notebook file contains a brief overview of the functionality of the main script.

To run the main script in terminal, simply type: ```python gaptopo_inference.py``` and press enter.
Two samplers will then run on the data; the PTM and then covPTM. Summary figures will be produced, and the status of the samplers will be printed to terminal.

To run the code, a working Python 2.7 distribution is necessary, and the newest version of these libraries installed:
	* itertools
	* random
	* copy
	* operator
	* scipy
	* numpy
	* pickle
	* matplotlib
