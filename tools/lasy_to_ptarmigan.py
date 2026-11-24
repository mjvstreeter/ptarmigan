#!/usr/bin/env python3
"""Export a propagated LASY pulse as a PTARMIGAN z-resolved volume."""

import argparse

import h5py
import numpy as np
from scipy.constants import c


def write_ptarmigan_volume(laser, z_positions, filename):
    """Write an xyt LASY laser at successive z planes.

    The LASY object must use Cartesian ``xyt`` geometry. The output stores
    the normalized vector-potential envelope in PTARMIGAN's version-1 volume
    schema. Its time axis is the retarded LASY time coordinate.
    """
    from lasy.backend import to_cpu
    from lasy.utils.laser_utils import field_to_vector_potential

    z_positions = np.asarray(z_positions, dtype=np.float64)
    if laser.dim != "xyt":
        raise ValueError("PTARMIGAN volume export currently requires LASY xyt geometry")
    if z_positions.ndim != 1 or z_positions.size < 2:
        raise ValueError("z_positions must contain at least two coordinates")
    if not np.all(np.isfinite(z_positions)) or np.any(np.diff(z_positions) <= 0):
        raise ValueError("z_positions must be finite and strictly increasing")

    omega = float(laser.profile.omega0)
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("LASY laser must have a finite positive carrier frequency")

    polarization = np.asarray(to_cpu(laser.profile.pol), dtype=np.complex128)
    if polarization.shape != (2,) or not np.isclose(np.linalg.norm(polarization), 1.0):
        raise ValueError("LASY polarization must be a normalized two-component vector")

    x, y, t = (np.asarray(to_cpu(axis), dtype=np.float64) for axis in laser.grid.axes)
    volume = np.empty((z_positions.size, y.size, x.size, t.size, 6), dtype=np.float64)
    current_z = float(laser.grid.position)

    for iz, z in enumerate(z_positions):
        if z < current_z:
            raise ValueError("z_positions must not precede the LASY grid's current position")
        laser.propagate(float(z - current_z))
        current_z = float(laser.grid.position)

        a = np.asarray(to_cpu(field_to_vector_potential(laser.grid, omega)))
        if a.shape != (x.size, y.size, t.size):
            raise ValueError("LASY returned an unexpected xyt field shape")
        ax = a * polarization[0]
        ay = a * polarization[1]
        az = 1j * (
            np.gradient(ax, x, axis=0, edge_order=2 if x.size > 2 else 1)
            + np.gradient(ay, y, axis=1, edge_order=2 if y.size > 2 else 1)
        ) / (omega / c)
        volume[iz, :, :, :, 0] = ax.real.transpose(1, 0, 2)
        volume[iz, :, :, :, 1] = ax.imag.transpose(1, 0, 2)
        volume[iz, :, :, :, 2] = ay.real.transpose(1, 0, 2)
        volume[iz, :, :, :, 3] = ay.imag.transpose(1, 0, 2)
        volume[iz, :, :, :, 4] = az.real.transpose(1, 0, 2)
        volume[iz, :, :, :, 5] = az.imag.transpose(1, 0, 2)

    with h5py.File(filename, "w") as output:
        group = output.create_group("ptarmigan_lasy")
        group.attrs["formatVersion"] = np.uint32(1)
        group.attrs["angularFrequency"] = omega
        group.attrs["polarization"] = np.array(
            [polarization[0].real, polarization[0].imag,
             polarization[1].real, polarization[1].imag],
            dtype=np.float64,
        )
        group.attrs["representation"] = "normalized_vector_potential_envelope"
        group.attrs["axisOrder"] = "z,y,x,t,component"
        group.attrs["componentOrder"] = "Ax.re,Ax.im,Ay.re,Ay.im,Az.re,Az.im"
        group.create_dataset("x", data=x)
        group.create_dataset("y", data=y)
        group.create_dataset("z", data=z_positions)
        group.create_dataset("t", data=t)
        group.create_dataset("a_envelope", data=volume, compression="gzip", shuffle=True)
        group["x"].attrs["unit"] = "m"
        group["y"].attrs["unit"] = "m"
        group["z"].attrs["unit"] = "m"
        group["t"].attrs["unit"] = "s"


def main():
    parser = argparse.ArgumentParser(
        description="Export an in-memory LASY xyt pulse to PTARMIGAN's z-volume format."
    )
    parser.add_argument("--help-api", action="store_true", help="show the Python API usage")
    args = parser.parse_args()
    if args.help_api:
        parser.exit(
            message=(
                "Import write_ptarmigan_volume from this module and call it with an "
                "initialized lasy.laser.Laser, an increasing z coordinate array (m), "
                "and an output .h5 path.\n"
            )
        )
    parser.error(
        "this utility is intended to be imported from the LASY pulse-generation script; "
        "use --help-api for the call signature"
    )


if __name__ == "__main__":
    main()
