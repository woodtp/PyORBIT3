"""Symmetric Enge quadrupole fringe profile."""

import math
from collections.abc import Iterable

from orbit.core.orbit_utils import Function

from .quad_field_profile import AbstractQuadFieldSourceFunction


class EngeFunction(AbstractQuadFieldSourceFunction):
    r"""
    Symmetric quadrupole fringe profile with default parameters from [1]_.

    The ideal, unnormalized Enge shape is

    .. math::

        u(z) = \frac{|z| - L/2}{D}, \qquad
        P(u) = \sum_{n=0}^{N} a_n u^n, \qquad
        g(z) = \frac{1}{1 + \exp\left[P(u(z))\right]}.

    Here :math:`z` is the longitudinal distance from the magnet center,
    :math:`L` is ``fringe_reference_separation``, and :math:`D` is
    ``fringe_scale_length``; all three lengths are in meters. Common choices
    for :math:`L` and :math:`D` are the magnet length and aperture diameter,
    respectively. Both parameters may also be fitted. The
    coefficients are dimensionless and ordered constant term first:
    ``[a0, a1, ..., aN]``. Reflection through :math:`|z|` uses the same
    single-edge fringe shape for the entrance and exit, located at
    :math:`z=-L/2` and :math:`z=+L/2`. At these positions,
    :math:`g(z)=1/(1+\exp(a_0))`.

    The implementation clips the exponent to
    ``[-_EXPONENT_CLIP_LIMIT, _EXPONENT_CLIP_LIMIT]`` and truncates the
    profile at a finite symmetric cutoff. ``relative_field_cutoff`` is the
    cutoff threshold relative to the unnormalized central value.
    ``getLimitsZ()`` returns the cutoff interval, and ``getFuncValue`` returns
    zero at and beyond its endpoints.

    Inside that interval, values use a piecewise-linear lookup table scaled
    to unit longitudinal integral by trapezoidal normalization of its sampled
    values. Thus the normalized profile has units of inverse meters; physical
    integrated quadrupole strength is supplied separately by the caller.

    Derivatives are evaluated analytically from the normalized, clipped
    underlying Enge profile. Reflection can create a cusp at :math:`z=0`,
    and truncation can create jumps at the cutoff. The methods return regular
    branch derivatives. At the center, the first derivative is set to zero
    and the second to the common one-sided curvature.
    At/beyond the cutoff and on clipped-exponent branches, derivatives are
    returned as zero. Derivative units are inverse meters squared and cubed
    for the first and second orders, respectively.

    References
    ----------
    .. [1] M. Berz, B. Erdélyi, and K. Makino, “Fringe field effects in small rings of large acceptance,” Phys. Rev. ST Accel. Beams, vol. 3, no. 12, p. 124001, Dec. 2000, doi: 10.1103/PhysRevSTAB.3.124001.

    """

    # Numerical saturation limit for the polynomial exponent, shared by values
    # and derivatives.
    _EXPONENT_CLIP_LIMIT = 30.0
    _DEFAULT_COEFFICIENTS = (0.296471, 4.533219, -2.270982, 1.068627, -0.036391, 0.022261,) # [a0, a1, a2, a3, a4, a5]

    def __init__(
        self,
        fringe_reference_separation: float,
        fringe_scale_length: float,
        relative_field_cutoff: float = 0.01,
        *,
        enge_coefficients: Iterable[float] | None = None,
        profile_table_points: int = 500,
    ) -> None:
        """Construct and normalize a symmetric Enge profile.

        Parameters
        ----------
        fringe_reference_separation : float
            Separation L of the entrance/exit reference planes, in meters.
            These planes lie at z = -L/2 and +L/2, where the corresponding
            fringe coordinate is zero. A common choice is the magnet length.
        fringe_scale_length : float
            Positive scale D in meters for u = (abs(z) - L/2) / D.
            A common choice is the magnet aperture diameter.
        relative_field_cutoff : float, optional
            Dimensionless threshold relative to the unnormalized central
            field, normally between zero and one. Defaults to 0.01.
            Determines the symmetric cutoff interval.
        enge_coefficients : iterable of float, optional
            Dimensionless polynomial coefficients [a0, a1, ...], constant
            term first. Copied and validated. None selects the Berz defaults.
            The profile must reach the threshold: the current cutoff search
            is unbounded for coefficients with non-decaying tails.
        profile_table_points : int, optional
            Number of samples from center to positive support boundary,
            including both endpoints. Must be at least two; defaults to 500.
            Also sets cutoff-search tolerance to search_step / table_points.

        Notes
        -----
        The field shape is normalized to unit longitudinal integral (1/m).
        ``field_half_extent`` and ``normalization_factor`` are derived during
        construction; the former may subsequently be set explicitly.
        """
        self._fringe_reference_separation = fringe_reference_separation
        self._fringe_scale_length = fringe_scale_length
        self._relative_field_cutoff = relative_field_cutoff

        self._enge_coefficients = self._validate_coefficients(
            self._DEFAULT_COEFFICIENTS if enge_coefficients is None else enge_coefficients
        )

        self._normalization_factor = 1.0
        self._profile_table_points = profile_table_points

        # -----find cut-off z value
        self._field_half_extent = self._findCutOff(fringe_scale_length, relative_field_cutoff)
        # ------------------------------------------------------

        self._profile_table = Function()
        self._normalize()

    @staticmethod
    def _validate_coefficients(coefficients: Iterable[float]) -> list[float]:
        if isinstance(coefficients, (str, bytes)):
            raise TypeError("coefficients must be an iterable of numbers")

        try:
            values = [float(value) for value in coefficients]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("coefficients must be an iterable of finite numbers") from exc

        if not values:
            raise ValueError("coefficients must not be empty")

        if not all(math.isfinite(value) for value in values):
            raise ValueError("coefficients must contain only finite numbers")

        return values

    def setEngeCoefficients(self, coefficients: Iterable[float]) -> None:
        """
        Sets new values for Enge function's coeffients.
        """
        values = self._validate_coefficients(coefficients)
        self._enge_coefficients = values
        step = self._fringe_reference_separation / 2
        self._field_half_extent = self._findCutOff(step, self._relative_field_cutoff)
        self._normalize()

    def setCutOffLevel(self, cutoff_level: float) -> None:
        """Sets the cutoff level for quad's  field"""
        step = self._fringe_reference_separation / 2
        self._relative_field_cutoff = cutoff_level
        self._field_half_extent = self._findCutOff(step, cutoff_level)
        self._normalize()

    def setCutOffZ(self, cutoff_z: float) -> None:
        """Sets the cutoff distance from the center of quad's field"""
        self._field_half_extent = cutoff_z
        self._normalize()

    def setLength(self, length: float) -> None:
        """Sets the length of quad's field"""
        self._fringe_reference_separation = length
        step = self._fringe_reference_separation / 2.0
        self._field_half_extent = self._findCutOff(step, self._relative_field_cutoff)
        self._normalize()

    def setAcceptanceDiameter(self, acceptance_diameter: float) -> None:
        """Sets the acceptance diameter of the quad"""
        self._fringe_scale_length = acceptance_diameter
        step = self._fringe_reference_separation / 2.0
        self._field_half_extent = self._findCutOff(step, self._relative_field_cutoff)
        self._normalize()

    def setNumberOfPoints(self, n_func_points: int) -> None:
        """Sets the number of points in the field function"""
        self._profile_table_points = n_func_points
        step = self._fringe_reference_separation / 2.0
        self._field_half_extent = self._findCutOff(step, self._relative_field_cutoff)
        self._normalize()

    def getCuttOffZ(self) -> float:
        """Returns the cutoff distance from the center of quad's field"""
        return self._field_half_extent

    def getNumberOfPoints(self) -> int:
        """Returns the number of points in the field function"""
        return self._profile_table_points

    def setFringeReferenceSeparation(self, value: float) -> None:
        """Set separation of the fringe reference planes in meters."""
        self.setLength(value)

    def setFringeScaleLength(self, value: float) -> None:
        """Set the Enge coordinate scale in meters."""
        self.setAcceptanceDiameter(value)

    def setRelativeFieldCutoff(self, value: float) -> None:
        """Set threshold relative to the central field."""
        self.setCutOffLevel(value)

    def setFieldHalfExtent(self, value: float) -> None:
        """Set the positive computational support boundary in meters."""
        self.setCutOffZ(value)

    def setProfileTablePoints(self, value: int) -> None:
        """Set samples on the positive half-profile, including endpoints."""
        self.setNumberOfPoints(value)

    def getFieldHalfExtent(self) -> float:
        """Return the positive computational support boundary in meters."""
        return self._field_half_extent

    def getProfileTablePoints(self) -> int:
        """Return samples on the positive half-profile, including endpoints."""
        return self._profile_table_points

    @property
    def fringe_reference_separation(self) -> float:
        """Separation of the fringe reference planes in meters."""
        return self._fringe_reference_separation

    @fringe_reference_separation.setter
    def fringe_reference_separation(self, value: float) -> None:
        self.setFringeReferenceSeparation(value)

    @property
    def fringe_scale_length(self) -> float:
        """Enge coordinate scale in meters, commonly the aperture diameter."""
        return self._fringe_scale_length

    @fringe_scale_length.setter
    def fringe_scale_length(self, value: float) -> None:
        self.setFringeScaleLength(value)

    @property
    def enge_coefficients(self) -> tuple[float, ...]:
        """Dimensionless polynomial coefficients, constant term first."""
        return tuple(self._enge_coefficients)

    @enge_coefficients.setter
    def enge_coefficients(self, value: Iterable[float]) -> None:
        self.setEngeCoefficients(value)

    @property
    def relative_field_cutoff(self) -> float:
        """Threshold relative to the central field value."""
        return self._relative_field_cutoff

    @relative_field_cutoff.setter
    def relative_field_cutoff(self, value: float) -> None:
        self.setRelativeFieldCutoff(value)

    @property
    def field_half_extent(self) -> float:
        """Positive computational support boundary in meters."""
        return self._field_half_extent

    @field_half_extent.setter
    def field_half_extent(self, value: float) -> None:
        self.setFieldHalfExtent(value)

    @property
    def profile_table_points(self) -> int:
        """Samples from center to positive support boundary, inclusive."""
        return self._profile_table_points

    @profile_table_points.setter
    def profile_table_points(self, value: int) -> None:
        self.setProfileTablePoints(value)

    @property
    def normalization_factor(self) -> float:
        """Derived unit-integral scale in inverse meters (read-only)."""
        return self._normalization_factor

    # Historical scalar properties retained for existing callers.
    @property
    def length(self) -> float:
        """Length parameter in meters."""
        return self._fringe_reference_separation

    @length.setter
    def length(self, value: float) -> None:
        self.setLength(value)

    @property
    def acceptance_diameter(self) -> float:
        """Acceptance diameter in meters."""
        return self._fringe_scale_length

    @acceptance_diameter.setter
    def acceptance_diameter(self, value: float) -> None:
        self.setAcceptanceDiameter(value)

    @property
    def cutoff_level(self) -> float:
        """Relative cutoff threshold."""
        return self._relative_field_cutoff

    @cutoff_level.setter
    def cutoff_level(self, value: float) -> None:
        self.setCutOffLevel(value)

    @property
    def cutoff_z(self) -> float:
        """Positive support limit in meters."""
        return self._field_half_extent

    @cutoff_z.setter
    def cutoff_z(self, value: float) -> None:
        self.setCutOffZ(value)

    @property
    def n_func_points(self) -> int:
        """Table size."""
        return self._profile_table_points

    @n_func_points.setter
    def n_func_points(self, value: int) -> None:
        self.setNumberOfPoints(value)

    @property
    def normalization(self) -> float:
        """Derived unit-integral scale, in inverse meters (read-only)."""
        return self._normalization_factor

    def _getTrueEngeFunc(self, x: float) -> float:
        """Returns the quad's field at the distance x from the center"""
        # x is the distance from the center of the magnet with the iron length l """
        x = (math.fabs(x) - self._fringe_reference_separation / 2.0) / self._fringe_scale_length
        sum_exp = self._enge_coefficients[0]
        x0 = x
        for i in range(1, len(self._enge_coefficients)):
            sum_exp += self._enge_coefficients[i] * x0
            x0 *= x
        if abs(sum_exp) > self._EXPONENT_CLIP_LIMIT:
            sum_exp = self._EXPONENT_CLIP_LIMIT * sum_exp / abs(sum_exp)
        return self._normalization_factor / (1.0 + math.exp(sum_exp))

    def _findCutOff(self, step: float, cutoff_level: float) -> float:
        """Finds the distance from the center where the field is less than cutoff level"""
        self._normalization_factor = 1.0
        init_val = self._getTrueEngeFunc(0.0)
        z = step
        val = self._getTrueEngeFunc(z) / init_val
        if val <= cutoff_level:
            return z
        while val > cutoff_level:
            z += step
            val = self._getTrueEngeFunc(z) / init_val
        z0 = z - step
        z1 = z
        step_z = step / self._profile_table_points
        while abs(z0 - z1) > step_z:
            z_new = (z0 + z1) / 2.0
            val_new = self._getTrueEngeFunc(z_new) / init_val
            if val_new <= cutoff_level:
                z1 = z_new
            else:
                z0 = z_new
        self._field_half_extent = (z0 + z1) / 2.0
        return self._field_half_extent

    def _normalize(self) -> None:
        """Normalizes the quad field function to the integral of 1"""
        self._normalization_factor = 1.0
        step = self._field_half_extent / (self._profile_table_points - 1)
        self._profile_table.clean()
        sum_int = 0.0
        for ind in range(self._profile_table_points):
            z = step * ind
            val = self._getTrueEngeFunc(z)
            self._profile_table.add(z, val)
            sum_int += val
        sum_int -= (self._getTrueEngeFunc(0.0) + self._getTrueEngeFunc(step * (self._profile_table_points - 1))) / 2.0
        sum_int *= 2.0 * step
        self._normalization_factor = 1.0 / sum_int
        self._profile_table.setConstStep(1)

    def getFuncValue(self, z: float) -> float:
        """Returns the quad's field at the distance z from the center"""
        if abs(z) >= self._profile_table.getMaxX():
            return 0.0
        return self._normalization_factor * self._profile_table.getY(abs(z))

    def _getEngeDerivatives(self, z: float) -> tuple[float, float]:
        """Return analytic first derivative and regular second derivative in z."""
        if abs(z) >= self._profile_table.getMaxX():
            return (0.0, 0.0)

        u = (abs(z) - self._fringe_reference_separation / 2.0) / self._fringe_scale_length

        # Horner evaluation of P(u), P'(u), and P''(u).
        polynomial = self._enge_coefficients[-1]
        first = 0.0
        second = 0.0
        for coefficient in reversed(self._enge_coefficients[:-1]):
            second = second * u + 2.0 * first
            first = first * u + polynomial
            polynomial = polynomial * u + coefficient

        # _getTrueEngeFunc clips the exponent, making these branches constant.
        if abs(polynomial) >= self._EXPONENT_CLIP_LIMIT:
            return (0.0, 0.0)

        exponential = math.exp(polynomial)
        denominator = 1.0 + exponential
        # exp(P)/(1+exp(P))**2 avoids cancellation when the field is near 1.
        factor = self._normalization_factor * exponential / denominator**2
        derivative = -factor * first / self._fringe_scale_length
        curvature = factor * ((exponential - 1.0) / denominator * first**2 - second) / self._fringe_scale_length**2
        if z < 0.0:
            derivative = -derivative
        elif z == 0.0:
            derivative = 0.0
        return (derivative, curvature)

    def getFuncDerivative(self, z: float) -> float:
        """Return the analytic first z derivative of the normalized Enge profile.

        This differentiates the underlying profile, not the linear lookup table
        used by getFuncValue. At z=0 return zero by symmetry, even if reflection
        creates a cusp. At/beyond the cutoff and on exponent-clipped branches
        return zero; discontinuities at branch boundaries are not differentiated.
        """
        return self._getEngeDerivatives(z)[0]

    def getFuncSecondDerivative(self, z: float) -> float:
        """Return the analytic regular second z derivative of the Enge profile.

        As with getFuncDerivative, this does not differentiate the lookup table
        or include jump terms. At z=0 return the common one-sided curvature;
        this is not a classical second derivative if the reflected profile has
        a cusp. At/beyond the cutoff and on exponent-clipped branches return zero.
        """
        return self._getEngeDerivatives(z)[1]

    def getFuncDerivatives(self, z: float) -> tuple[float, float]:
        """Return (first derivative, second derivative)."""
        return self._getEngeDerivatives(z)

    def getLimitsZ(self) -> tuple[float, float]:
        """Returns the tuple with min and max Z value for this field"""
        z_max = self._profile_table.getMaxX()
        return -z_max, z_max

