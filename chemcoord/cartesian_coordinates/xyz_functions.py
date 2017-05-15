from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
try:
    # import itertools.imap as map
    import itertools.izip as zip
except ImportError:
    pass
import numpy as np
import pandas as pd
import collections
from threading import Thread
import subprocess
import os
import tempfile
import warnings
from chemcoord.cartesian_coordinates.cartesian_class_main import Cartesian
from chemcoord import export
from chemcoord.configuration import settings
import io
from io import open
import re
import textwrap

# from pkg_resources import get_distribution, DistributionNotFound
# __version__ = get_distribution(__name__).version
# try:
#     __version__ = get_distribution(__name__).version
# except DistributionNotFound:
#    # package is not installed
#    pass


def view(molecule, viewer=settings['defaults']['viewer'], use_curr_dir=False):
    """View your molecule or list of molecules.

    .. note:: This function writes a temporary file and opens it with
        an external viewer.
        If you modify your molecule afterwards you have to recall view
        in order to see the changes.

    Args:
        molecule: Can be a cartesian, or a list of cartesians.
        viewer (str): The external viewer to use. The default is
            specified in settings.viewer
        use_curr_dir (bool): If True, the temporary file is written to
            the current diretory. Otherwise it gets written to the
            OS dependendent temporary directory.

    Returns:
        None:
    """
    try:
        molecule.view(viewer=viewer, use_curr_dir=use_curr_dir)
    except AttributeError:
        if pd.api.types.is_list_like(molecule):
            cartesian_list = molecule
        else:
            raise ValueError('Argument is neither list nor Cartesian.')
        if use_curr_dir:
            TEMP_DIR = os.path.curdir
        else:
            TEMP_DIR = tempfile.gettempdir()

        def give_filename(i):
            filename = 'ChemCoord_list_' + str(i) + '.molden'
            return os.path.join(TEMP_DIR, filename)

        i = 1
        while os.path.exists(give_filename(i)):
            i = i + 1

        to_molden(cartesian_list, buf=give_filename(i))

        def open_file(i):
            """Open file and close after being finished."""
            try:
                subprocess.check_call([viewer, give_filename(i)])
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise
            finally:
                if use_curr_dir:
                    pass
                else:
                    os.remove(give_filename(i))
        Thread(target=open_file, args=(i,)).start()


def to_molden(cartesian_list, buf=None, sort_index=True,
              overwrite=True, float_format='{:.6f}'.format):
    """Write a list of Cartesians into a molden file.

    .. note:: Since it permamently writes a file, this function
        is strictly speaking **not sideeffect free**.
        The list to be written is of course not changed.

    Args:
        cartesian_list (list):
        buf (str): StringIO-like, optional buffer to write to
        sort_index (bool): If sort_index is true, the Cartesian
            is sorted by the index before writing.
        overwrite (bool): May overwrite existing files.
        float_format (one-parameter function): Formatter function
            to apply to column’s elements if they are floats.
            The result of this function must be a unicode string.

    Returns:
        formatted : string (or unicode, depending on data and options)
    """
    if sort_index:
        cartesian_list = [molecule.sort_index() for molecule in cartesian_list]

    give_header = ("[MOLDEN FORMAT]\n"
                   + "[N_GEO]\n"
                   + str(len(cartesian_list)) + "\n"
                   + '[GEOCONV]\n'
                   + 'energy\n{energy}'
                   + 'max-force\n{max_force}'
                   + 'rms-force\n{rms_force}'
                   + '[GEOMETRIES] (XYZ)\n').format

    values = len(cartesian_list) * '1\n'
    header = give_header(energy=values, max_force=values, rms_force=values)

    coordinates = [x.to_xyz(sort_index=sort_index, float_format=float_format)
                   for x in cartesian_list]
    output = header + '\n'.join(coordinates)

    if buf is not None:
        if overwrite:
            with open(buf, mode='w') as f:
                f.write(output)
        else:
            with open(buf, mode='x') as f:
                f.write(output)
    else:
        return output


def write_molden(*args, **kwargs):
    """Deprecated, use :func:`~chemcoord.xyz_functions.to_molden`
    """
    message = 'Will be removed in the future. Please use to_molden().'
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(message, DeprecationWarning)
    return to_molden(*args, **kwargs)


def from_molden(inputfile, start_index=0, get_bonds=True):
    """Read a molden file.

    Args:
        inputfile (str):
        start_index (int):

    Returns:
        list: A list containing :class:`~chemcoord.Cartesian` is returned.
    """
    f = open(inputfile, 'r')

    found = False
    while not found:
        line = f.readline()
        if line.strip() == '[N_GEO]':
            found = True
            number_of_molecules = int(f.readline().strip())

    found = False
    while not found:
        line = f.readline()
        if line.strip() == '[GEOMETRIES] (XYZ)':
            found = True
            current_line = f.tell()
            number_of_atoms = int(f.readline().strip())
            f.seek(current_line)

    list_of_cartesians = []
    for i in range(number_of_molecules):
        molecule_in = [f.readline()
                       for j in range(number_of_atoms + 2)]
        molecule_in = ''.join(molecule_in)
        molecule_in = io.StringIO(molecule_in)
        molecule = Cartesian.from_xyz(molecule_in,
                                      start_index=start_index,
                                      get_bonds=get_bonds)
        list_of_cartesians.append(molecule)

    f.close()
    return list_of_cartesians


def read_molden(*args, **kwargs):
    """Deprecated, use :func:`~chemcoord.xyz_functions.from_molden`
    """
    message = 'Will be removed in the future. Please use from_molden().'
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(message, DeprecationWarning)
    return from_molden(*args, **kwargs)


def isclose(a, b, align=True, rtol=1.e-5, atol=1.e-8):
    """Compare two molecules for numerical equality.

    Args:
        a (Cartesian):
        b (Cartesian):
        align (bool): a and b are
            prealigned along their principal axes of inertia and moved to their
            barycenters before comparing.
        rtol (float): Relative tolerance for the numerical equality comparison
            look into :func:`numpy.isclose` for further explanation.
        atol (float): Relative tolerance for the numerical equality comparison
            look into :func:`numpy.isclose` for further explanation.

    Returns:
        bool:
    """
    pretest = (set(a.index) == set(b.index)
               and np.alltrue(a.loc[:, 'atom'] == b.loc[a.index, 'atom']))

    if align and pretest:
        A = a.inertia()['transformed_Cartesian'].location()
        B = b.inertia()['transformed_Cartesian'].loc[a.index, :].location()
        return np.allclose(A, B, rtol=rtol, atol=atol)
    elif pretest:
        A = a.location()
        B = b.loc[a.index, :].location()
        return np.allclose(A, B, rtol=rtol, atol=atol)
    else:
        return False