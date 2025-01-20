"""Collection of classes for dealing with SDE models"""

from typing import Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class SDEModel(ABC):
    """Abstract class for dealing with SDE Models methods of the form

    dr_t = a(r_t, t)dt + b(r_t, t)dW_t
    """

    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def calibrate_with_data(cls, data: pd.DataFrame) -> "SDEModel":
        """Initialize model with parameters calibrated by provided data.

        Args:
            data: pandas.DataFrame with two columns -- 'date' and 'rate'
        """
        raise NotImplementedError(
            f"calibrate_from_data is not implemented for {cls.__name__}"
        )

    @abstractmethod
    def a(self, x: float, t: float) -> float:
        """a(r_t, t) function from the SDE"""
        raise NotImplementedError(f"a is not implemented for {type(self).__name__}")

    @abstractmethod
    def b(self, x: float, t: float) -> float:
        """b(r_t, t) function from the SDE"""
        raise NotImplementedError(f"b is not implemented for {type(self).__name__}")

    @abstractmethod
    def b_prime(self, x: float) -> float:
        """Derivative of b(x) from the SDE. Only for autonomous SDE."""
        raise NotImplementedError(f"b_prime is not implemented for {type(self).__name__}")

    def _simulate_euler_maruyama(
        self, x_0: float, t: float, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Helper method for `simulate`, see `help(SDEModel.simulate)`
        for more details


        Simulation using Euler-Maruyama scheme
        """
        simulation = np.zeros(n + 1)
        # not using np.arrange for precision reasons
        time_points = np.linspace(0, t, num=n + 1, endpoint=True)
        simulation[0] = x_0
        for n_, t_ in enumerate(time_points[1:], 1):
            dt = t_ - time_points[n_ - 1]
            prev_value = simulation[n_ - 1]
            simulation[n_] = (
                simulation[n_ - 1]
                + self.a(prev_value, t_) * dt
                + self.b(prev_value, t_) * np.random.normal(loc=0.0, scale=dt)
            )
        return time_points, simulation

    def _simulate_milstein(
        self, x_0, t: float, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Helper method for `simulate`, see `help(SDEModel.simulate)`
        for more details

        Simulation using Milstein scheme. Only for autonomous SDE
        """
        simulation = np.zeros(n + 1)
        # not using np.arrange for precision reasons
        simulation[0] = x_0
        time_points = np.linspace(0, t, num=n + 1, endpoint=True)
        assert len(time_points) == n + 1
        for n_, t_ in enumerate(time_points[1:], 1):
            dt = t_ - time_points[n_ - 1]
            prev_value = simulation[n_ - 1]
            dw_n = np.random.normal(loc=0.0, scale=dt)
            b = self.b(prev_value, t_)
            simulation[n_] = (
                simulation[n_ - 1]
                + self.a(prev_value, t_) * dt
                + b * dw_n
                + 0.5 * b * self.b_prime(prev_value) * (dw_n * dw_n - dt)
            )
        return time_points, simulation

    def simulate(
        self, x_0: float, t: float, n: int, method: str = "euler_maruyama"
    ) -> np.array:
        """Simulate the SDE.

        Args:
            x_0: initial value
            t: simulation will be on the time interval [0, t]
            n: [0, t] will be partitioned into n subintervals
            method: either "euler_maruyama" or "milstein"
        """
        if method == "euler_maruyama":
            return self._simulate_euler_maruyama(x_0, t, n)
        if method == "milstein":
            return self._simulate_milstein(x_0, t, n)
        raise ValueError(f"Invalid method: {method}")

    def plot_simulation( # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        x_0: float,
        t: float,
        n: int,
        method: str = "euler_maruyama",
        sim_num: int = 1,
        title: Optional[str] = None,
    ) -> None:
        """Plot simulation.

        Args:
            x_0, t, n, method: see SDEModel.simulate
            sim_num: number of simulations.
            title: title of the graph. If None, it will be automatically generated.
        """
        for _ in range(sim_num):
            plt.plot(*self.simulate(x_0, t, n, method))
        plt.xlabel("time (days)")
        plt.ylabel("interest rate")
        if title is None:
            title = (
                f"Simulation of {self.__class__.__name__} model, using {method} scheme"
            )
        plt.title(title)
        plt.show()


class Vasicek(SDEModel):
    """Vasicek short-rate model defined as

    dr_t = alpha(beta - r_t)dt + sigma * dW_t
    """

    def __init__(self, alpha: float, beta: float, sigma: float):
        """alpha, beta and sigma parameters of the SDE"""
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    @classmethod
    def calibrate_with_data(cls, data: pd.DataFrame) -> "Vasicek":
        """See SDEModel.calibrate_with_data"""
        y = data.rate[1:].to_numpy() - data.rate[:-1].to_numpy()
        x = data.rate[:-1].to_numpy().reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        alpha = -reg.coef_[0]
        beta = reg.intercept_ / alpha
        sigma_array = y - alpha * (beta - x.reshape(-1))
        sigma = np.std(sigma_array)
        return Vasicek(alpha, beta, sigma)

    def a(self, x: float, _t: float) -> float:
        """See SDEModel.a"""
        return self.alpha * (self.beta - x)

    def b(self, _x: float, _t: float) -> float:
        """See SDEModel.b"""
        return self.sigma

    def b_prime(self, _x: float) -> float:
        """See SDEModel.b_prime"""
        return 0


class CIR(SDEModel):
    """Cox–Ingersoll–Ross short-rate model defined as

    dr_t = alpha(beta - r_t)dt + sigma * sqrt(r_t)dW_t
    """

    def __init__(self, alpha: float, beta: float, sigma: float):
        """alpha, beta and sigma parameters from the SDE"""
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    def a(self, x: float, _t: float) -> float:
        """See SDEModel.a"""
        return self.alpha * (self.beta - x)

    def b(self, x: float, _t: float) -> float:
        """See SDEModel.b"""
        return self.sigma * np.sqrt(x)

    def b_prime(self, x: float) -> float:
        """See SDEModel.b_prime"""

    @classmethod
    def calibrate_with_data(cls, data: pd.DataFrame):
        """See SDEModel.calibrate_with_data"""
        y = data.rate[1:].to_numpy() - data.rate[:-1].to_numpy()
        x = data.rate[:-1].to_numpy().reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        alpha = -reg.coef_[0]
        beta = reg.intercept_ / alpha
        sigma_array = (y - alpha * (beta - x.reshape(-1))) / np.sqrt(x.reshape(-1))
        sigma = np.std(sigma_array)
        return CIR(alpha, beta, sigma)


class RB(SDEModel):
    """Rendleman–Bartter short-rate model defined as:

    dr_t = theta * r_tdt + sigma * r_t dW_t
    """

    def __init__(self, theta: float, sigma: float):
        """theta and sigma parameters from the SDE"""
        self.theta = theta
        self.sigma = sigma

    def a(self, x: float, _t: float) -> float:
        """See SDEModel.a"""
        return self.theta * x

    def b(self, x: float, _t: float) -> float:
        """See SDEModel.b"""
        return self.sigma * x

    def b_prime(self, x: float) -> float:
        """See SDEModel.b_prime"""

    @classmethod
    def calibrate_with_data(cls, data: pd.DataFrame):
        """See SDEModel.calibrate_with_data"""
        y = data.rate[1:].to_numpy() - data.rate[:-1].to_numpy()
        x = data.rate[:-1].to_numpy().reshape(-1, 1)
        reg = LinearRegression(fit_intercept=False).fit(x, y)
        theta = reg.coef_[0]
        sigma_array = (y - theta * x.reshape(-1)) / x.reshape(-1)
        sigma = np.std(sigma_array)
        return RB(theta, sigma)
