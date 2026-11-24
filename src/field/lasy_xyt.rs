use std::error::Error;
use std::f64::consts;
use std::fmt;
use std::sync::Arc;

use hdf5::File;

use crate::constants::*;
use crate::geometry::{FourVector, ThreeVector};

use super::{Envelope, Field, LaserParameters, Polarization};

const GROUP_PATH: &str = "ptarmigan_lasy";

#[derive(Clone)]
pub struct LasyXytField {
    params: LaserParameters,
    omega: f64,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    t: Vec<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
    nt: usize,
    // [z, y, x, t, (Ax.re, Ax.im, Ay.re, Ay.im, Az.re, Az.im)]
    a_envelope: Arc<Vec<f64>>,
}

#[derive(Debug)]
pub struct LasyFieldError(String);

impl fmt::Display for LasyFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for LasyFieldError {}

impl LasyXytField {
    /// Loads a PTARMIGAN LASY volume (see docs/input_guide/laser.md).
    pub fn from_hdf5(path: &str) -> Result<Self, LasyFieldError> {
        let file = File::open(path).map_err(|e| LasyFieldError(e.to_string()))?;
        let group = file
            .group(GROUP_PATH)
            .map_err(|e| LasyFieldError(e.to_string()))?;

        let version = group
            .attr("formatVersion")
            .and_then(|attr| attr.read_scalar::<u32>())
            .map_err(|e| LasyFieldError(format!("missing or invalid formatVersion: {}", e)))?;
        if version != 1 {
            return Err(LasyFieldError(format!(
                "unsupported volume format version {}",
                version
            )));
        }

        let omega = group
            .attr("angularFrequency")
            .and_then(|attr| attr.read_scalar::<f64>())
            .map_err(|e| LasyFieldError(format!("missing or invalid angularFrequency: {}", e)))?;
        if !omega.is_finite() || omega <= 0.0 {
            return Err(LasyFieldError(
                "angularFrequency must be finite and positive".to_owned(),
            ));
        }

        let polarization = group
            .attr("polarization")
            .and_then(|attr| attr.read_raw::<f64>())
            .map_err(|e| LasyFieldError(format!("missing or invalid polarization: {}", e)))?;
        if polarization.len() != 4 || polarization.iter().any(|v| !v.is_finite()) {
            return Err(LasyFieldError(
                "polarization must contain [px.re, px.im, py.re, py.im]".to_owned(),
            ));
        }

        let read_axis = |name: &str| -> Result<Vec<f64>, LasyFieldError> {
            group
                .dataset(name)
                .and_then(|dataset| dataset.read_raw::<f64>())
                .map_err(|e| LasyFieldError(format!("cannot read {} axis: {}", name, e)))
        };
        let x = read_axis("x")?;
        let y = read_axis("y")?;
        let z = read_axis("z")?;
        let t = read_axis("t")?;

        for (name, axis) in [("x", &x), ("y", &y), ("z", &z), ("t", &t)] {
            Self::validate_axis(name, axis)?;
        }

        let dataset = group
            .dataset("a_envelope")
            .map_err(|e| LasyFieldError(format!("cannot read a_envelope dataset: {}", e)))?;
        let expected_shape = [z.len(), y.len(), x.len(), t.len(), 6];
        if dataset.shape() != expected_shape {
            return Err(LasyFieldError(format!(
                "a_envelope must have shape [nz, ny, nx, nt, 6] = {:?}, got {:?}",
                expected_shape,
                dataset.shape()
            )));
        }
        let a_envelope = dataset
            .read_raw::<f64>()
            .map_err(|e| LasyFieldError(format!("cannot read a_envelope data: {}", e)))?;
        if a_envelope.iter().any(|v| !v.is_finite()) {
            return Err(LasyFieldError(
                "a_envelope contains non-finite values".to_owned(),
            ));
        }

        let a0 = a_envelope
            .chunks_exact(6)
            .map(|a| {
                (a[0].hypot(a[1]))
                    .hypot(a[2].hypot(a[3]))
                    .hypot(a[4].hypot(a[5]))
            })
            .fold(0.0, f64::max);
        if a0 == 0.0 {
            return Err(LasyFieldError("a_envelope is zero everywhere".to_owned()));
        }

        let px = f64::hypot(polarization[0], polarization[1]);
        let py = f64::hypot(polarization[2], polarization[3]);
        if (px.hypot(py) - 1.0).abs() > 1.0e-3 {
            return Err(LasyFieldError(
                "polarization vector must be normalized".to_owned(),
            ));
        }
        let relative_phase = polarization[0] * polarization[3] - polarization[1] * polarization[2];
        let circular = (px - py).abs() < 1.0e-3 && relative_phase.abs() > 0.49;
        let pol = if circular {
            Polarization::Circular
        } else {
            Polarization::Linear
        };
        let pol_angle = if pol == Polarization::Linear {
            polarization[2].atan2(polarization[0])
        } else {
            0.0
        };
        let wavelength = 2.0 * consts::PI * SPEED_OF_LIGHT / omega;
        let params = LaserParameters {
            a0,
            wavelength,
            pol,
            pol_angle,
            focusing: true,
            waist: f64::INFINITY,
            envelope: Envelope::Gaussian,
            n_cycles: SPEED_OF_LIGHT * (t[t.len() - 1] - t[0]) / wavelength,
            chirp_b: 0.0,
        };

        Ok(Self {
            params,
            omega,
            nx: x.len(),
            ny: y.len(),
            nz: z.len(),
            nt: t.len(),
            x,
            y,
            z,
            t,
            a_envelope: Arc::new(a_envelope),
        })
    }

    pub fn params(&self) -> LaserParameters {
        self.params
    }

    fn validate_axis(name: &str, axis: &[f64]) -> Result<(), LasyFieldError> {
        if axis.len() < 2 || axis.iter().any(|v| !v.is_finite()) {
            return Err(LasyFieldError(format!(
                "{} axis must contain at least two finite values",
                name
            )));
        }
        let step = axis[1] - axis[0];
        if step <= 0.0
            || axis.windows(2).any(|w| {
                let delta = w[1] - w[0];
                !delta.is_finite() || (delta - step).abs() > 1.0e-8 * step.abs()
            })
        {
            return Err(LasyFieldError(format!(
                "{} axis must be uniformly spaced and increasing",
                name
            )));
        }
        Ok(())
    }

    fn axis_step(axis: &[f64]) -> f64 {
        axis[1] - axis[0]
    }

    fn bracket(axis: &[f64], value: f64) -> Option<(usize, f64)> {
        if value < axis[0] || value > axis[axis.len() - 1] {
            return None;
        }
        if value == axis[axis.len() - 1] {
            return Some((axis.len() - 2, 1.0));
        }
        let step = Self::axis_step(axis);
        let index = ((value - axis[0]) / step).floor() as usize;
        Some((index, (value - axis[index]) / step))
    }

    fn envelope_at(&self, x: f64, y: f64, z: f64, t: f64) -> Option<[f64; 6]> {
        let (ix, fx) = Self::bracket(&self.x, x)?;
        let (iy, fy) = Self::bracket(&self.y, y)?;
        let (iz, fz) = Self::bracket(&self.z, z)?;
        let (it, ft) = Self::bracket(&self.t, t)?;
        let mut result = [0.0; 6];

        for dz in 0..=1 {
            for dy in 0..=1 {
                for dx in 0..=1 {
                    for dt in 0..=1 {
                        let weight = if dx == 0 { 1.0 - fx } else { fx }
                            * if dy == 0 { 1.0 - fy } else { fy }
                            * if dz == 0 { 1.0 - fz } else { fz }
                            * if dt == 0 { 1.0 - ft } else { ft };
                        let base = (((((iz + dz) * self.ny + (iy + dy)) * self.nx + (ix + dx))
                            * self.nt
                            + (it + dt))
                            * 6) as usize;
                        for component in 0..6 {
                            result[component] += weight * self.a_envelope[base + component];
                        }
                    }
                }
            }
        }
        Some(result)
    }

    fn physical_potential(&self, r: FourVector) -> Option<ThreeVector> {
        let tau = (r[0] - r[3]) / SPEED_OF_LIGHT;
        let a = self.envelope_at(r[1], r[2], r[3], tau)?;
        let (sin, cos) = (-self.omega * tau).sin_cos();
        let ax = a[0] * cos - a[1] * sin;
        let ay = a[2] * cos - a[3] * sin;
        let az = a[4] * cos - a[5] * sin;
        Some(ThreeVector::new(ax, ay, az))
    }

    fn derivative(
        center: ThreeVector,
        minus: Option<ThreeVector>,
        plus: Option<ThreeVector>,
        delta: f64,
    ) -> ThreeVector {
        match (minus, plus) {
            (Some(minus), Some(plus)) => (plus - minus) / (2.0 * delta),
            (Some(minus), None) => (center - minus) / delta,
            (None, Some(plus)) => (plus - center) / delta,
            (None, None) => [0.0; 3].into(),
        }
    }

    fn offset(r: FourVector, component: usize, delta: f64) -> FourVector {
        match component {
            0 => FourVector::new(r[0], r[1] + delta, r[2], r[3]),
            1 => FourVector::new(r[0], r[1], r[2] + delta, r[3]),
            2 => FourVector::new(r[0], r[1], r[2], r[3] + delta),
            _ => FourVector::new(r[0] + SPEED_OF_LIGHT * delta, r[1], r[2], r[3]),
        }
    }

    fn envelope_amplitude(&self, r: FourVector) -> f64 {
        let tau = (r[0] - r[3]) / SPEED_OF_LIGHT;
        self.envelope_at(r[1], r[2], r[3], tau)
            .map(|a| {
                (a[0].hypot(a[1]))
                    .hypot(a[2].hypot(a[3]))
                    .hypot(a[4].hypot(a[5]))
            })
            .unwrap_or(0.0)
    }
}

impl Field for LasyXytField {
    fn max_timestep(&self) -> Option<f64> {
        Some(
            0.25 * [
                Self::axis_step(&self.x) / SPEED_OF_LIGHT,
                Self::axis_step(&self.y) / SPEED_OF_LIGHT,
                Self::axis_step(&self.z) / SPEED_OF_LIGHT,
                Self::axis_step(&self.t),
            ]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        )
    }

    fn contains(&self, r: FourVector) -> bool {
        let tau = (r[0] - r[3]) / SPEED_OF_LIGHT;
        Self::bracket(&self.x, r[1]).is_some()
            && Self::bracket(&self.y, r[2]).is_some()
            && Self::bracket(&self.z, r[3]).is_some()
            && Self::bracket(&self.t, tau).is_some()
    }

    fn ideal_initial_z(&self) -> f64 {
        self.z[self.nz - 1].max(0.0) + (-SPEED_OF_LIGHT * self.t[0]).max(0.0)
    }

    fn fields(&self, r: FourVector) -> (ThreeVector, ThreeVector, f64) {
        let Some(a) = self.physical_potential(r) else {
            return ([0.0; 3].into(), [0.0; 3].into(), 0.0);
        };
        let dx = Self::axis_step(&self.x);
        let dy = Self::axis_step(&self.y);
        let dz = Self::axis_step(&self.z);
        let dt = Self::axis_step(&self.t);

        let derivative_x = Self::derivative(
            a,
            self.physical_potential(Self::offset(r, 0, -dx)),
            self.physical_potential(Self::offset(r, 0, dx)),
            dx,
        );
        let derivative_y = Self::derivative(
            a,
            self.physical_potential(Self::offset(r, 1, -dy)),
            self.physical_potential(Self::offset(r, 1, dy)),
            dy,
        );
        let derivative_z = Self::derivative(
            a,
            self.physical_potential(Self::offset(r, 2, -dz)),
            self.physical_potential(Self::offset(r, 2, dz)),
            dz,
        );
        let derivative_t = Self::derivative(
            a,
            self.physical_potential(Self::offset(r, 3, -dt)),
            self.physical_potential(Self::offset(r, 3, dt)),
            dt,
        );

        let potential_to_field = ELECTRON_MASS * SPEED_OF_LIGHT / ELEMENTARY_CHARGE;
        let e = -potential_to_field * derivative_t;
        let b = potential_to_field
            * ThreeVector::new(
                derivative_y[2] - derivative_z[1],
                derivative_z[0] - derivative_x[2],
                derivative_x[1] - derivative_y[0],
            );
        (e, b, self.envelope_amplitude(r))
    }

    fn energy(&self) -> (f64, &'static str) {
        let iz = self.nz / 2;
        let dx = Self::axis_step(&self.x);
        let dy = Self::axis_step(&self.y);
        let dt = Self::axis_step(&self.t);
        let mut norm_sqd = 0.0;
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                for it in 0..self.nt {
                    let base = ((((iz * self.ny + iy) * self.nx + ix) * self.nt + it) * 6) as usize;
                    norm_sqd += self.a_envelope[base].powi(2)
                        + self.a_envelope[base + 1].powi(2)
                        + self.a_envelope[base + 2].powi(2)
                        + self.a_envelope[base + 3].powi(2)
                        + self.a_envelope[base + 4].powi(2)
                        + self.a_envelope[base + 5].powi(2);
                }
            }
        }
        let electric = ELECTRON_MASS * SPEED_OF_LIGHT * self.omega / ELEMENTARY_CHARGE;
        let energy =
            0.5 * VACUUM_PERMITTIVITY * SPEED_OF_LIGHT * electric.powi(2) * norm_sqd * dx * dy * dt;
        (energy, "J")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_metadata_and_reconstructs_plane_wave_fields() {
        let path = std::env::temp_dir().join(format!("ptarmigan-lasy-{}.h5", std::process::id()));
        let path_str = path.to_str().unwrap();
        let omega = 0.5 * consts::PI;
        let x = [-1.0, 0.0, 1.0];
        let y = [-1.0, 0.0, 1.0];
        let z = [-0.1 * SPEED_OF_LIGHT, 0.0, 0.1 * SPEED_OF_LIGHT];
        let t: Vec<f64> = (0..=200).map(|i| (i as f64 - 100.0) * 0.1).collect();
        let shape = [z.len(), y.len(), x.len(), t.len(), 6];
        let mut a = vec![0.0; shape.iter().product()];
        for component in a.chunks_exact_mut(6) {
            component[0] = 1.0;
        }

        {
            let file = File::create(path_str).unwrap();
            let group = file.create_group(GROUP_PATH).unwrap();
            group
                .new_attr::<u32>()
                .create("formatVersion")
                .unwrap()
                .write_scalar(&1)
                .unwrap();
            group
                .new_attr::<f64>()
                .create("angularFrequency")
                .unwrap()
                .write_scalar(&omega)
                .unwrap();
            group
                .new_attr::<f64>()
                .shape([4])
                .create("polarization")
                .unwrap()
                .write_raw(&[1.0, 0.0, 0.0, 0.0])
                .unwrap();
            for (name, axis) in [("x", &x[..]), ("y", &y[..]), ("z", &z[..]), ("t", &t[..])] {
                group
                    .new_dataset_builder()
                    .with_data(axis)
                    .create(name)
                    .unwrap();
            }
            group
                .new_dataset::<f64>()
                .shape(shape)
                .create("a_envelope")
                .unwrap()
                .write_raw(&a)
                .unwrap();
        }

        let field = LasyXytField::from_hdf5(path_str).unwrap();
        assert!((field.params().wavelength - 4.0 * SPEED_OF_LIGHT).abs() < 1.0e-12);
        assert!((field.params().a0 - 1.0).abs() < 1.0e-12);

        let r = FourVector::new(SPEED_OF_LIGHT, 0.0, 0.0, 0.0);
        let (e, b, _) = field.fields(r);
        let expected_e =
            ELECTRON_MASS * SPEED_OF_LIGHT / ELEMENTARY_CHARGE * (omega * 0.1).sin() / 0.1;
        assert!((e[0] - expected_e).abs() / expected_e < 1.0e-12);
        assert!(
            (b[1] - expected_e / SPEED_OF_LIGHT).abs() / (expected_e / SPEED_OF_LIGHT) < 1.0e-12
        );

        drop(field);
        std::fs::remove_file(path).unwrap();
    }
}
