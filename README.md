# AmericanOptionEarlyExerciseBoundary
This project is about calculating high precision American Option Early Exercise Boundaries for the Black-Scholes-Modell,
where 
* $dS = (r-q) S dt + \sigma dW_t$  under Q
* $dB = r B dt$

The underlying algorithm is based on the paper
"High Performance American Option Pricing, 2015, written by Leif Andersen, Mark Lake and Dimitri Offengenden,
Bank of America Merrill Lynch". All code in Python.

### Algorithm in a nutshell:
1. Compute Chebyshev nodes $x_i$ and establishes the collocation grid $\tau_i = x_i^2$. Initialize Chebyshev Interpolation.

2. On above spacing compute the QD_plus approximation of the Early Exercise boundaries ("Analytical Approximations for the
Critical Stock Prices of American
Options: A Performance Comparison", Minqiang Li, Li
Georgia Institute of Technology) therefore just solving a root finding problem via ridders method/secant method.
   
3. Starting from the QD_plus approximation of the boundary, calculate a new approximation on the collocation grid 
via fixpoint iteration
    *  $B_{plus}(tau) = B(tau) - \eta (B(tau) - \Phi(B(tau))) $

    where $B(tau)$ is obtained via Chebyshev Interpolation on the transformed boundary space H, with
    * $H(B) = log(B / K)^2 $
    
Algorithm stops after certain number of fixpoint iterations or if the changes of B_plus induced by fixpoint iteration 
are smaller than stop_by_diff=1e-6.

## Whats included?
* American Exercise Boundary calculator (Call / Put)
* QD_plus Early Exercise Boundary calculator (Call / Put)
* American Option Pricer (Call / Put) for given boundary
* American Option Premium (Call / Put) for given boundary
* European Option Pricer (Call / Put)
* Chebyshev Interpolator

## How to use?
#### Import:
```
from Option_Solver import Option_solver

```
#### Create Instance:
```
r, q, sigma, K, T, option_type = 0.05, 0.1, 0.25, 100, 2, 'Put'

option = Option_Solver.Option_Solver(r, q, sigma, K, T, option_type)
```

#### Calculate Boundary:
```
option.create_boundary()

```
#### Calculate Prices:

```
S, tau = 120, 1  # tau has to be in [0, T]

American_price = option.American_price(S, tau) 
American_premium = option.premium(S, tau)
European_price = European_price(S, tau)
```




