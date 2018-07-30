# -*- coding: utf-8 -*-
import re
from io import StringIO

import attr
from chemcoord import Cartesian, Zmat
from chemcoord.internal_coordinates.zmat_functions import TestOperators
from numpy import isclose
from pandas import DataFrame, read_csv
from scipy.constants import physical_constants
from sympy import Symbol


@attr.s(repr=False, cmp=False)
class CartesianVibration:
    reference = attr.ib(type=Cartesian)
    data = attr.ib(type=DataFrame)
    dsplcmnt = attr.ib(type=dict)

    @staticmethod
    def _partition_molden(filepath):
        is_section_key = re.compile('\[(.*)\]').search
        relevant = {'MOLDEN FORMAT', 'Atoms', 'N_GEO', 'GEOCONV', 'GEOMETRIES',
                    'FREQ', 'FR-COORD', 'FR-NORM-COORD', 'INT'}
        sections = {}
        with open(filepath, 'r') as f:
            line = f.readline()
            current_section = None
            while line:
                match = is_section_key(line)
                if match:
                    current_section = match.group(1)
                    sections[current_section] = ''
                elif current_section in relevant:
                    sections[current_section] += line
                line = f.readline()
        return sections

    @staticmethod
    def _get_molecule(reference, chunk):
        positions = DataFrame(
            [row.split() for row in chunk], columns=['x', 'y', 'z'],
            index=reference.index, dtype='f8')
        cols = ['atom', 'x', 'y', 'z']
        return Cartesian(positions.join(reference.loc[:, 'atom']).loc[:, cols])

    @classmethod
    def read_molden(cls, filepath, start_index=0, exclude_trans_rot=True):
        sections = cls._partition_molden(filepath)

        to_Ang = physical_constants['Bohr radius'][0] * 1e10
        data = DataFrame(
            [l.strip() for l in sections['FREQ'].split('\n') if l],
            columns=['frequency'], dtype='f8')
        if 'INT' in sections:
            data.loc[:, 'intensity'] = sections['INT']
        if start_index:
            data.index = range(start_index, start_index + len(data))

        def get_chunks(text, L):
            splitted = text.split('\n')
            n = 1
            while n + L <= len(splitted):
                yield splitted[n:n + L]
                n += L + 1

        names = ['atom', 'x', 'y', 'z']
        df = read_csv(StringIO(sections['FR-COORD']),
                      delim_whitespace=True, names=names)
        reference = Cartesian(df) * to_Ang
        chunks = get_chunks(sections['FR-NORM-COORD'], len(reference))

        dsplcmnt = {}
        for i, chunk in zip(data.index, chunks):
            if exclude_trans_rot and not isclose(data.loc[i, 'frequency'], 0):
                dsplcmnt[i] = (cls._get_molecule(reference, chunk)
                               * Symbol(f't{i}')
                               * to_Ang)
        if exclude_trans_rot:
            data = data[~isclose(data.loc[:, 'frequency'], 0)]
        return cls(reference, data, dsplcmnt)

    def __repr__(self):
        return f'{len(self.data)} vibrational modes in cartesian coordinates.'


@attr.s(repr=False, cmp=False)
class ZmatVibration:
    reference = attr.ib(type=Zmat)
    data = attr.ib(type=DataFrame)
    dsplcmnt = attr.ib(type=dict)

    @classmethod
    def from_cart_vib(cls, cart_vib: CartesianVibration, const_table=None):
        ref = cart_vib.reference
        if const_table is None:
            z_ref = ref.get_zmat()
            const_table = z_ref.loc[:, ['b', 'a', 'd']]
        else:
            z_ref = ref.get_zmat(const_table)

        z_dsplcmnt = {}
        for i, displacement in cart_vib.dsplcmnt.items():
            distorted = displacement.subs(Symbol(f't{i}'), 1) + ref
            z_distorted = distorted.get_zmat(const_table)
            with TestOperators(False):
                z_dsplcmnt[i] = ((z_distorted - z_ref).minimize_dihedrals()
                                 * Symbol(f's{i}'))
        return cls(z_ref, cart_vib.data, z_dsplcmnt)

    def __repr__(self):
        return f'{len(self.data)} vibrational modes in Z-matrix coordinates.'
