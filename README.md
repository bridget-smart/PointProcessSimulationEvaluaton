This is flexible tool to simulate data representing $N$ associated point processes, where the coordination structure is dictated by an underlying coordination network and user input determines the type of association between the pairs of processes. The flexibility of this data simulation framework allows for empirical evaluation of methods which produce a measure of association between two point processes. 

The functions to be evaluated can be provided in evaluation_framework.py, and the default and iterable values of the simulation framework can be provided. For each set of parameter values, a simulated dataset is generated and each possible association measure is validated against ground truth values. These are set by default to test against directly coordinating behaviours and similarity between generating processes, however different ground truth values can be specified by altering or replacing the `generating_similarity_matrix' function given in functions/evaluating_methods.

To recreate the plots and results within the associated work see `making_plots.ipynb'. This work also provides further details for both the data simulation procedure and the evaluation framework.

