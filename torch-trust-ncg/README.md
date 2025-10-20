# Torch-TrustNCG

Pytorch implementation of a Trust Region Newton Conjugate Gradient method. 

## Installation

To install the model please follow the next steps in the specified order:
1. Clone this repository and install it using the *setup.py* script: 
```Shell
git clone https://github.com/vchoutas/torch-trust-ncg.git
```
2. If you do not wish to modify the optimizer then run:
```Shell
python setup.py install
```
1. If you want to be able to modify the optimizer then run:
```Shell
python setup.py build develop
```

## Usage

To create the optimizer simply run:
```Python
    optimizer = TrustRegion(parameter_list)
```
where paremeter_list is the list of parameters you wish to optimize. To perform
one optimization step simply call the step function and pass a closure that
computes the loss and the gradients. Note the the closure should have a boolean
argument named *backward*, so that the optimizer avoids unnecessary backward
passes.

For a simple example see the __main__.py function. To run it for the rosenbrock
function execute the following command:
```Shell
python -m torchtrustncg
```

## Citation

For more details see chapter 7.2 of "Numerical Optimization, Nocedal and
Wright":

```
@Book{NoceWrig06,
    Title                    = {Numerical Optimization},
    Author                   = {Jorge Nocedal and Stephen J. Wright},
    Publisher                = {Springer},
    Year                     = {2006},
    Address                  = {New York, NY, USA},
    Edition                  = {second}
}
```
