"""Abstract interface for normalized longitudinal quadrupole profiles."""

from abc import ABC, abstractmethod


class AbstractQuadFieldSourceFunction(ABC):
    """
    It is an abstract class describing the quadrupole magnetic
    field as a function of the longitudinal coordinate.
    """

    @abstractmethod
    def getLimitsZ(self) -> tuple[float, float]:
        ...

    @abstractmethod
    def getFuncValue(self, z) -> float:
        """
        Returns the quad's normalized field distribution
        at the distance z from the center.
        the distribution should be normalized to 1 over the
        integration along the longitudinal coordinate.
        """
        ...

    @abstractmethod
    def getFuncDerivative(self, z) -> float:
        """
        Returns the derivative of the getFuncValue(z)
        """
        ...

    def getFuncSecondDerivative(self, z: float) -> float:
        """Returns the second derivative of getFuncValue(z); defaults to zero."""
        return 0.0

    def getFuncDerivatives(self, z: float) -> tuple[float, float]:
        """Returns the first and second derivatives of getFuncValue(z)."""
        return (
            self.getFuncDerivative(z),
            self.getFuncSecondDerivative(z),
        )
