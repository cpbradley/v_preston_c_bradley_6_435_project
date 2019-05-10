README for the final project for 6.435. Prepared by Chris Bradley and Victoria Preston

An implementation of an adjusted inference algorithm in:
Inference in the Space of Topological Maps: An MCMC-based Approach
https://smartech.gatech.edu/bitstream/handle/1853/38451/Ranganathan04iros.pdf

To run the code in terminal, simply type in: python gaptopo_inference.py and press enter.
Two samplers will then run on the data. First, the sampler from the paper, then our modified version.

Parameters can be set in the main function at the bottom of the document (lines 703 to 708).
Can also set the world you would like to perform inference over by changing line 680.
Options include simulation_example_1, simulation_example_2, and adversarial_world (current setting).
