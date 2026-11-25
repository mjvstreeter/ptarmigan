use num_complex::Complex;
use std::f64::consts;

use crate::constants::*;
use crate::field::{FastFocusedLaser, Field, Polarization};
use crate::geometry::{FourVector, ThreeVector};

#[derive(Copy, Clone)]
struct FrequencyComponent {
    k: f64,
    waist: f64,
    rayleigh_range: f64,
    focus: f64,
    weight: Complex<f64>,
}

/// A paraxial, linearly polarized focused pulse built from a Gaussian spectrum.
///
/// The waist of each frequency component is set by the same f-number. Its
/// frequency-dependent focus and quadratic spectral phase model longitudinal
/// chromatic aberration and group-delay dispersion, respectively.
pub struct SpectralFocusedLaser {
    a0: f64,
    wavelength: f64,
    fwhm_duration: f64,
    omega: f64,
    sigma: f64,
    f_number: f64,
    beam_fill: f64,
    gdd: f64,
    lca: f64,
    n_sigma: f64,
    pol_angle: f64,
    max_phase: f64,
    components: Vec<FrequencyComponent>,
}

impl SpectralFocusedLaser {
    #[allow(unused)]
    pub fn new(
        a0: f64,
        wavelength: f64,
        fwhm_duration: f64,
        f_number: f64,
        gdd: f64,
        lca: f64,
        beam_fill: f64,
        n_freq: usize,
        n_sigma: f64,
        pol_angle: f64,
    ) -> Self {
        assert!(
            n_freq >= 2,
            "The spectral pulse requires at least two frequency samples."
        );
        assert!(
            wavelength > 0.0 && fwhm_duration > 0.0,
            "Wavelength and pulse duration must be positive."
        );
        assert!(
            f_number > 0.0 && beam_fill > 0.0,
            "The f-number and beam fill must be positive."
        );
        assert!(n_sigma > 0.0, "The spectral bandwidth must be positive.");

        let omega = 2.0 * consts::PI * SPEED_OF_LIGHT / wavelength;
        let sigma = 2.0 * consts::LN_2.sqrt() / fwhm_duration;
        assert!(
            omega - n_sigma * sigma > 0.0,
            "The requested spectral bandwidth includes non-positive frequencies."
        );

        let n_eff = f_number / beam_fill;
        let delta_omega = 2.0 * n_sigma * sigma / ((n_freq - 1) as f64);
        let spectral_weights: Vec<(f64, f64)> = (0..n_freq)
            .map(|j| {
                let offset = -n_sigma * sigma + (j as f64) * delta_omega;
                let weight = (-offset.powi(2) * fwhm_duration.powi(2) / (8.0 * consts::LN_2)).exp();
                (offset, weight)
            })
            .collect();
        let normalization: f64 = spectral_weights.iter().map(|(_, weight)| weight).sum();

        let components = spectral_weights
            .into_iter()
            .map(|(offset, spectral_weight)| {
                let frequency = omega + offset;
                let k = frequency / SPEED_OF_LIGHT;
                let waist = 4.0 * SPEED_OF_LIGHT * n_eff / frequency;
                let rayleigh_range = 0.5 * k * waist.powi(2);
                let focus = lca * offset;
                let weight = spectral_weight * Complex::new(0.0, 0.5 * gdd * offset.powi(2)).exp()
                    / normalization;
                FrequencyComponent {
                    k,
                    waist,
                    rayleigh_range,
                    focus,
                    weight,
                }
            })
            .collect();

        let tau_out = fwhm_duration
            * (1.0 + (4.0 * consts::LN_2 * gdd / fwhm_duration.powi(2)).powi(2)).sqrt();
        let max_phase =
            3.0 * omega * tau_out + (omega / SPEED_OF_LIGHT) * lca.abs() * n_sigma * sigma;

        Self {
            a0,
            wavelength,
            fwhm_duration,
            omega,
            sigma,
            f_number,
            beam_fill,
            gdd,
            lca,
            n_sigma,
            pol_angle,
            max_phase,
            components,
        }
    }

    fn scalar_potential_and_grad(&self, r: FourVector) -> (f64, [f64; 4]) {
        let r = r.rotate_around_z(-self.pol_angle);
        let (x, y, z, ct) = (r[1], r[2], r[3], r[0]);
        let radius_sqd = x.powi(2) + y.powi(2);
        let mut potential = Complex::new(0.0, 0.0);
        let mut gradient = [Complex::new(0.0, 0.0); 4];

        for component in &self.components {
            let q = Complex::new(1.0, (z - component.focus) / component.rayleigh_range);
            let q_grad = Complex::new(0.0, 1.0 / component.rayleigh_range);
            let radial = (-radius_sqd / component.waist.powi(2)) / q;
            let phase = Complex::new(0.0, component.k * (z - ct));
            let term = component.weight * (Complex::new(1.0, 0.0) / q) * (radial + phase).exp();

            potential += term;
            gradient[0] += term * Complex::new(0.0, -component.k);
            gradient[1] += term * (-2.0 * x / (component.waist.powi(2) * q));
            gradient[2] += term * (-2.0 * y / (component.waist.powi(2) * q));

            let dlog_dz = -q_grad / q
                + radius_sqd * q_grad / (component.waist.powi(2) * q.powi(2))
                + Complex::new(0.0, component.k);
            gradient[3] += term * dlog_dz;
        }

        (
            self.a0 * potential.re,
            [
                self.a0 * gradient[0].re,
                self.a0 * gradient[1].re,
                self.a0 * gradient[2].re,
                self.a0 * gradient[3].re,
            ],
        )
    }

    /// Returns the normalized spatial vector potential `e A / (m_e c)`.
    #[allow(dead_code)]
    pub fn vector_potential(&self, r: FourVector) -> ThreeVector {
        let (potential, _) = self.scalar_potential_and_grad(r);
        ThreeVector::new(potential, 0.0, 0.0).rotate_around_z(self.pol_angle)
    }

    /// Returns the gradient of the normalized vector potential with respect
    /// to `(ct, x, y, z)`, with each entry containing the three spatial
    /// components of the corresponding derivative.
    pub fn grad_vector_potential(&self, r: FourVector) -> [ThreeVector; 4] {
        let (_, grad) = self.scalar_potential_and_grad(r);
        let (sin, cos) = self.pol_angle.sin_cos();
        [
            ThreeVector::new(grad[0], 0.0, 0.0).rotate_around_z(self.pol_angle),
            ThreeVector::new(cos * grad[1] - sin * grad[2], 0.0, 0.0)
                .rotate_around_z(self.pol_angle),
            ThreeVector::new(sin * grad[1] + cos * grad[2], 0.0, 0.0)
                .rotate_around_z(self.pol_angle),
            ThreeVector::new(grad[3], 0.0, 0.0).rotate_around_z(self.pol_angle),
        ]
    }
}

impl Field for SpectralFocusedLaser {
    fn max_timestep(&self) -> Option<f64> {
        let max_omega = self.omega + self.n_sigma * self.sigma;
        let multiplier = (3_f64.sqrt() / (5.0 * ALPHA_FINE * self.a0)).min(0.1);
        Some(multiplier / max_omega)
    }

    fn contains(&self, r: FourVector) -> bool {
        let phase = (self.omega / SPEED_OF_LIGHT) * (r[3] - r[0]);
        phase > -self.max_phase
    }

    fn ideal_initial_z(&self) -> f64 {
        2.0 * SPEED_OF_LIGHT
            * self.fwhm_duration
            * (1.0 + (4.0 * consts::LN_2 * self.gdd / self.fwhm_duration.powi(2)).powi(2)).sqrt()
            + self.lca.abs() * self.n_sigma * self.sigma
    }

    #[allow(non_snake_case)]
    fn fields(&self, r: FourVector) -> (ThreeVector, ThreeVector, f64) {
        let grad = self.grad_vector_potential(r);
        let E = -(ELECTRON_MASS * SPEED_OF_LIGHT_SQD / ELEMENTARY_CHARGE) * grad[0];
        let B = (ELECTRON_MASS * SPEED_OF_LIGHT / ELEMENTARY_CHARGE)
            * ThreeVector::new(
                grad[2][2] - grad[3][1],
                grad[3][0] - grad[1][2],
                grad[1][1] - grad[2][0],
            );
        let a =
            ELEMENTARY_CHARGE * E.norm_sqr().sqrt() / (ELECTRON_MASS * SPEED_OF_LIGHT * self.omega);
        (E, B, a)
    }

    fn energy(&self) -> (f64, &'static str) {
        let waist = 2.0 * self.wavelength * self.f_number / (consts::PI * self.beam_fill);
        let n_cycles = SPEED_OF_LIGHT * self.fwhm_duration / self.wavelength;
        FastFocusedLaser::new(
            self.a0,
            self.wavelength,
            waist,
            n_cycles,
            Polarization::Linear,
            self.pol_angle,
        )
        .with_envelope(super::Envelope::Gaussian)
        .energy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn potential_and_gradient_are_consistent() {
        let laser = SpectralFocusedLaser::new(
            1.0, 0.8e-6, 30.0e-15, 7.0, 200.0e-30, -3.4e-18, 1.0, 65, 4.5, 0.37,
        );
        let r = FourVector::new(0.3e-6, 1.1e-6, -0.7e-6, 15.0e-6);
        let grad = laser.grad_vector_potential(r);
        let step = 1.0e-11;

        for axis in 0..4 {
            let mut plus = [r[0], r[1], r[2], r[3]];
            let mut minus = plus;
            plus[axis] += step;
            minus[axis] -= step;
            let numerical = (laser.vector_potential(plus.into())
                - laser.vector_potential(minus.into()))
                / (2.0 * step);
            let error =
                (numerical - grad[axis]).norm_sqr().sqrt() / grad[axis].norm_sqr().sqrt().max(1.0);
            assert!(
                error < 1.0e-6,
                "gradient error in coordinate {}: {}",
                axis,
                error
            );
        }
    }

    #[test]
    fn spectrum_is_normalized_at_focus_without_chirp_or_aberration() {
        let laser =
            SpectralFocusedLaser::new(0.8, 0.8e-6, 30.0e-15, 7.0, 0.0, 0.0, 1.0, 129, 4.5, 0.0);
        let potential = laser.vector_potential(FourVector::new(0.0, 0.0, 0.0, 0.0));
        assert!((potential[0] - 0.8).abs() < 1.0e-14);
        assert!(potential[1].abs() < 1.0e-14);
        assert!(potential[2].abs() < 1.0e-14);
    }
}
