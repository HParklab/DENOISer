import os
import sys
import numpy as np
import torch
from scipy.spatial import cKDTree
from typing import Union
import glob
from pathlib import Path
from typing import Tuple

from .constants import AtomicWeightsDecimal


AA_to_tip = {
    "ALA": "CB",
    "CYS": "SG",
    "ASP": "CG",
    "ASN": "CG",
    "GLU": "CD",
    "GLN": "CD",
    "PHE": "CZ",
    "HIS": "NE2",
    "ILE": "CD1",
    "GLY": "CA",
    "LEU": "CG",
    "MET": "SD",
    "ARG": "CZ",
    "LYS": "NZ",
    "PRO": "CG",
    "VAL": "CB",
    "TYR": "OH",
    "TRP": "CH2",
    "SER": "OG",
    "THR": "OG1",
}

AMINOACID = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


AA_ONE_LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

NON_STANDARD_AA = {
    "PYR": "TYR",
    "Y1P": "TYR",
    "SEP": "SER",
    "S1P": "SER",
    "TPO": "THR",
    "T1P": "THR",
    "MSE": "MET",
    "PTR": "TYR",
    "PYL": "LYS",
    "MLY": "LYS",
    "HYP": "PRO",
    "CSO": "CYS",
    "CSS": "CYS",
    "KCX": "LYS",
    "CSD": "ALA",
}


residuemap = dict([(AMINOACID[i], i) for i in range(len(AMINOACID))])

ALL_AAS = ["UNK"] + AMINOACID

N_AATYPE = len(ALL_AAS)

aa2short = {
    "ALA": (" N  ", " CA ", " C  ", " CB ", None, None, None, None),
    "ARG": (" N  ", " CA ", " C  ", " CB ", " CG ", " CD ", " NE ", " CZ "),
    "ASN": (" N  ", " CA ", " C  ", " CB ", " CG ", " OD1", None, None),
    "ASP": (" N  ", " CA ", " C  ", " CB ", " CG ", " OD1", None, None),
    "CYS": (" N  ", " CA ", " C  ", " CB ", " SG ", None, None, None),
    "GLN": (" N  ", " CA ", " C  ", " CB ", " CG ", " CD ", " OE1", None),
    "GLU": (" N  ", " CA ", " C  ", " CB ", " CG ", " CD ", " OE1", None),
    "GLY": (" N  ", " CA ", " C  ", None, None, None, None, None),
    "HIS": (" N  ", " CA ", " C  ", " CB ", " CG ", " ND1", None, None),
    "ILE": (" N  ", " CA ", " C  ", " CB ", " CG1", " CD1", None, None),
    "LEU": (" N  ", " CA ", " C  ", " CB ", " CG ", " CD1", None, None),
    "LYS": (" N  ", " CA ", " C  ", " CB ", " CG ", " CD ", " CE ", " NZ "),
    "MET": (" N  ", " CA ", " C  ", " CB ", " CG ", " SD ", " CE ", None),
    "PHE": (" N  ", " CA ", " C  ", " CB ", " CG ", " CD1", None, None),
    "PRO": (" N  ", " CA ", " C  ", " CB ", " CG ", " CD ", None, None),
    "SER": (" N  ", " CA ", " C  ", " CB ", " OG ", None, None, None),
    "THR": (" N  ", " CA ", " C  ", " CB ", " OG1", None, None, None),
    "TRP": (" N  ", " CA ", " C  ", " CB ", " CG ", " CD1", None, None),
    "TYR": (" N  ", " CA ", " C  ", " CB ", " CG ", " CD1", None, None),
    "VAL": (" N  ", " CA ", " C  ", " CB ", " CG1", None, None, None),
}


# Atom types:
atypes = {
    ("ALA", "CA"): "CAbb",
    ("ALA", "CB"): "CH3",
    ("ALA", "C"): "CObb",
    ("ALA", "N"): "Nbb",
    ("ALA", "O"): "OCbb",
    ("ARG", "CA"): "CAbb",
    ("ARG", "CB"): "CH2",
    ("ARG", "C"): "CObb",
    ("ARG", "CD"): "CH2",
    ("ARG", "CG"): "CH2",
    ("ARG", "CZ"): "aroC",
    ("ARG", "NE"): "Narg",
    ("ARG", "NH1"): "Narg",
    ("ARG", "NH2"): "Narg",
    ("ARG", "N"): "Nbb",
    ("ARG", "O"): "OCbb",
    ("ASN", "CA"): "CAbb",
    ("ASN", "CB"): "CH2",
    ("ASN", "C"): "CObb",
    ("ASN", "CG"): "CNH2",
    ("ASN", "ND2"): "NH2O",
    ("ASN", "N"): "Nbb",
    ("ASN", "OD1"): "ONH2",
    ("ASN", "O"): "OCbb",
    ("ASP", "CA"): "CAbb",
    ("ASP", "CB"): "CH2",
    ("ASP", "C"): "CObb",
    ("ASP", "CG"): "COO",
    ("ASP", "N"): "Nbb",
    ("ASP", "OD1"): "OOC",
    ("ASP", "OD2"): "OOC",
    ("ASP", "O"): "OCbb",
    ("CYS", "CA"): "CAbb",
    ("CYS", "CB"): "CH2",
    ("CYS", "C"): "CObb",
    ("CYS", "N"): "Nbb",
    ("CYS", "O"): "OCbb",
    ("CYS", "SG"): "S",
    ("GLN", "CA"): "CAbb",
    ("GLN", "CB"): "CH2",
    ("GLN", "C"): "CObb",
    ("GLN", "CD"): "CNH2",
    ("GLN", "CG"): "CH2",
    ("GLN", "NE2"): "NH2O",
    ("GLN", "N"): "Nbb",
    ("GLN", "OE1"): "ONH2",
    ("GLN", "O"): "OCbb",
    ("GLU", "CA"): "CAbb",
    ("GLU", "CB"): "CH2",
    ("GLU", "C"): "CObb",
    ("GLU", "CD"): "COO",
    ("GLU", "CG"): "CH2",
    ("GLU", "N"): "Nbb",
    ("GLU", "OE1"): "OOC",
    ("GLU", "OE2"): "OOC",
    ("GLU", "O"): "OCbb",
    ("GLY", "CA"): "CAbb",
    ("GLY", "C"): "CObb",
    ("GLY", "N"): "Nbb",
    ("GLY", "O"): "OCbb",
    ("HIS", "CA"): "CAbb",
    ("HIS", "CB"): "CH2",
    ("HIS", "C"): "CObb",
    ("HIS", "CD2"): "aroC",
    ("HIS", "CE1"): "aroC",
    ("HIS", "CG"): "aroC",
    ("HIS", "ND1"): "Nhis",
    ("HIS", "NE2"): "Ntrp",
    ("HIS", "N"): "Nbb",
    ("HIS", "O"): "OCbb",
    ("ILE", "CA"): "CAbb",
    ("ILE", "CB"): "CH1",
    ("ILE", "C"): "CObb",
    ("ILE", "CD1"): "CH3",
    ("ILE", "CG1"): "CH2",
    ("ILE", "CG2"): "CH3",
    ("ILE", "N"): "Nbb",
    ("ILE", "O"): "OCbb",
    ("LEU", "CA"): "CAbb",
    ("LEU", "CB"): "CH2",
    ("LEU", "C"): "CObb",
    ("LEU", "CD1"): "CH3",
    ("LEU", "CD2"): "CH3",
    ("LEU", "CG"): "CH1",
    ("LEU", "N"): "Nbb",
    ("LEU", "O"): "OCbb",
    ("LYS", "CA"): "CAbb",
    ("LYS", "CB"): "CH2",
    ("LYS", "C"): "CObb",
    ("LYS", "CD"): "CH2",
    ("LYS", "CE"): "CH2",
    ("LYS", "CG"): "CH2",
    ("LYS", "N"): "Nbb",
    ("LYS", "NZ"): "Nlys",
    ("LYS", "O"): "OCbb",
    ("MET", "CA"): "CAbb",
    ("MET", "CB"): "CH2",
    ("MET", "C"): "CObb",
    ("MET", "CE"): "CH3",
    ("MET", "CG"): "CH2",
    ("MET", "N"): "Nbb",
    ("MET", "O"): "OCbb",
    ("MET", "SD"): "S",
    ("PHE", "CA"): "CAbb",
    ("PHE", "CB"): "CH2",
    ("PHE", "C"): "CObb",
    ("PHE", "CD1"): "aroC",
    ("PHE", "CD2"): "aroC",
    ("PHE", "CE1"): "aroC",
    ("PHE", "CE2"): "aroC",
    ("PHE", "CG"): "aroC",
    ("PHE", "CZ"): "aroC",
    ("PHE", "N"): "Nbb",
    ("PHE", "O"): "OCbb",
    ("PRO", "CA"): "CAbb",
    ("PRO", "CB"): "CH2",
    ("PRO", "C"): "CObb",
    ("PRO", "CD"): "CH2",
    ("PRO", "CG"): "CH2",
    ("PRO", "N"): "Npro",
    ("PRO", "O"): "OCbb",
    ("SER", "CA"): "CAbb",
    ("SER", "CB"): "CH2",
    ("SER", "C"): "CObb",
    ("SER", "N"): "Nbb",
    ("SER", "OG"): "OH",
    ("SER", "O"): "OCbb",
    ("THR", "CA"): "CAbb",
    ("THR", "CB"): "CH1",
    ("THR", "C"): "CObb",
    ("THR", "CG2"): "CH3",
    ("THR", "N"): "Nbb",
    ("THR", "OG1"): "OH",
    ("THR", "O"): "OCbb",
    ("TRP", "CA"): "CAbb",
    ("TRP", "CB"): "CH2",
    ("TRP", "C"): "CObb",
    ("TRP", "CD1"): "aroC",
    ("TRP", "CD2"): "aroC",
    ("TRP", "CE2"): "aroC",
    ("TRP", "CE3"): "aroC",
    ("TRP", "CG"): "aroC",
    ("TRP", "CH2"): "aroC",
    ("TRP", "CZ2"): "aroC",
    ("TRP", "CZ3"): "aroC",
    ("TRP", "NE1"): "Ntrp",
    ("TRP", "N"): "Nbb",
    ("TRP", "O"): "OCbb",
    ("TYR", "CA"): "CAbb",
    ("TYR", "CB"): "CH2",
    ("TYR", "C"): "CObb",
    ("TYR", "CD1"): "aroC",
    ("TYR", "CD2"): "aroC",
    ("TYR", "CE1"): "aroC",
    ("TYR", "CE2"): "aroC",
    ("TYR", "CG"): "aroC",
    ("TYR", "CZ"): "aroC",
    ("TYR", "N"): "Nbb",
    ("TYR", "OH"): "OH",
    ("TYR", "O"): "OCbb",
    ("VAL", "CA"): "CAbb",
    ("VAL", "CB"): "CH1",
    ("VAL", "C"): "CObb",
    ("VAL", "CG1"): "CH3",
    ("VAL", "CG2"): "CH3",
    ("VAL", "N"): "Nbb",
    ("VAL", "O"): "OCbb",
}

# Atome type to index
atype2num = {
    "CNH2": 0,
    "Npro": 1,
    "CH1": 2,
    "CH3": 3,
    "CObb": 4,
    "aroC": 5,
    "OOC": 6,
    "Nhis": 7,
    "Nlys": 8,
    "COO": 9,
    "NH2O": 10,
    "S": 11,
    "Narg": 12,
    "OCbb": 13,
    "Ntrp": 14,
    "Nbb": 15,
    "CH2": 16,
    "CAbb": 17,
    "ONH2": 18,
    "OH": 19,
}

gentype2num = {
    "CS": 0,
    "CS1": 1,
    "CS2": 2,
    "CS3": 3,
    "CD": 4,
    "CD1": 5,
    "CD2": 6,
    "CR": 7,
    "CT": 8,
    "CSp": 9,
    "CDp": 10,
    "CRp": 11,
    "CTp": 12,
    "CST": 13,
    "CSQ": 14,
    "HO": 15,
    "HN": 16,
    "HS": 17,
    # Nitrogen
    "Nam": 18,
    "Nam2": 19,
    "Nad": 20,
    "Nad3": 21,
    "Nin": 22,
    "Nim": 23,
    "Ngu1": 24,
    "Ngu2": 25,
    "NG3": 26,
    "NG2": 27,
    "NG21": 28,
    "NG22": 29,
    "NG1": 30,
    "Ohx": 31,
    "Oet": 32,
    "Oal": 33,
    "Oad": 34,
    "Oat": 35,
    "Ofu": 36,
    "Ont": 37,
    "OG2": 38,
    "OG3": 39,
    "OG31": 40,
    # S/P
    "Sth": 41,
    "Ssl": 42,
    "SR": 43,
    "SG2": 44,
    "SG3": 45,
    "SG5": 46,
    "PG3": 47,
    "PG5": 48,
    # Halogens
    "Br": 49,
    "I": 50,
    "F": 51,
    "Cl": 52,
    "BrR": 53,
    "IR": 54,
    "FR": 55,
    "ClR": 56,
    # Metals
    "Ca2p": 57,
    "Mg2p": 58,
    "Mn": 59,
    "Fe2p": 60,
    "Fe3p": 60,
    "Zn2p": 61,
    "Co2p": 62,
    "Cu2p": 63,
    "Cd": 64,
}

gentype2atom = {
    "CS": "C",
    "CS1": "C",
    "CS2": "C",
    "CS3": "C",
    "CD": "C",
    "CD1": "C",
    "CD2": "C",
    "CR": "C",
    "CT": "C",
    "CSp": "C",
    "CDp": "C",
    "CRp": "C",
    "CTp": "C",
    "CST": "C",
    "CSQ": "C",
    "HO": "H",
    "HN": "H",
    "HS": "H",
    # Nitrogen
    "Nam": "N",
    "Nam2": "N",
    "Nad": "N",
    "Nad3": "N",
    "Nin": "N",
    "Nim": "N",
    "Ngu1": "N",
    "Ngu2": "N",
    "NG3": "N",
    "NG2": "N",
    "NG21": "N",
    "NG22": "N",
    "NG1": "N",
    "Ohx": "O",
    "Oet": "O",
    "Oal": "O",
    "Oad": "O",
    "Oat": "O",
    "Ofu": "O",
    "Ont": "O",
    "OG2": "O",
    "OG3": "O",
    "OG31": "O",
    # S/P
    "Sth": "S",
    "Ssl": "S",
    "SR": "S",
    "SG2": "S",
    "SG3": "S",
    "SG5": "S",
    "PG3": "P",
    "PG5": "P",
    # Halogens
    "Br": "Br",
    "I": "I",
    "F": "F",
    "Cl": "Cl",
    "BrR": "Br",
    "IR": "I",
    "FR": "F",
    "ClR": "Cl",
    # Metals
    "Ca2p": "Ca",
    "Mg2p": "Mg",
    "Mn": "Mn",
    "Fe2p": "Fe",
    "Fe3p": "Fe",
    "Zn2p": "Zn",
    "Co2p": "Co",
    "Cu2p": "Cu",
    "Cd": "Cd",
}


gentype2simple = {
    "CS": 0,
    "CS1": 0,
    "CS3": 0,
    "CST": 0,
    "CSQ": 0,
    "CSp": 0,
    "CD": 1,
    "CD1": 1,
    "CD2": 1,
    "CDp": 1,
    "CT": 2,
    "CTp": 2,
    "CR": 3,
    "CRp": 3,
    "HN": 4,
    "HO": 4,
    "HS": 4,
    "Nam": 5,
    "Nam2": 5,
    "NG3": 5,
    "Nad": 6,
    "Nad3": 6,
    "Nin": 6,
    "Nim": 6,
    "Ngu1": 6,
    "Ngu2": 6,
    "NG2": 6,
    "NG21": 6,
    "NG22": 6,
    "NG1": 7,
    "Ohx": 8,
    "OG3": 8,
    "Oet": 8,
    "OG31": 8,
    "Oal": 9,
    "Oad": 9,
    "Oat": 9,
    "Ofu": 9,
    "Ont": 9,
    "OG2": 9,
    "Sth": 10,
    "Ssl": 10,
    "SR": 10,
    "SG2": 10,
    "SG3": 10,
    "SG5": 10,
    "PG3": 11,
    "PG5": 11,
    "F": 12,
    "Cl": 13,
    "Br": 14,
    "I": 15,
    "FR": 12,
    "ClR": 13,
    "BrR": 14,
    "IR": 15,
    "Ca2p": 16,
    "Mg2p": 17,
    "Mn": 18,
    "Fe2p": 19,
    "Fe3p": 19,
    "Zn2p": 20,
    "Co2p": 21,
    "Cu2p": 22,
    "Cd": 23,
}


AAprop = {
    "netq": [0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    "nchi": [0, 1, 2, 2, 3, 3, 2, 2, 3, 0, 3, 4, 4, 4, 1, 1, 2, 2, 1, 1],
    "Kappa": [
        (5.000, 2.250, 2.154),
        (11.000, 6.694, 5.141),
        (8.000, 3.938, 3.746),
        (8.000, 3.938, 4.660),
        (6.000, 3.200, 3.428),
        (9.000, 4.840, 4.639),
        (9.000, 4.840, 5.592),
        (4.000, 3.000, 2.879),
        (8.100, 4.000, 2.381),
        (8.000, 3.938, 3.841),
        (8.000, 3.938, 3.841),
        (9.000, 6.125, 5.684),
        (8.000, 5.143, 5.389),
        (9.091, 4.793, 3.213),
        (5.143, 2.344, 1.661),
        (6.000, 3.200, 2.809),
        (7.000, 3.061, 2.721),
        (10.516, 4.680, 2.737),
        (10.083, 4.889, 3.324),
        (6.000, 1.633, 1.567),
    ],
}


def find_gentype2num(at):
    if at in gentype2num:
        return gentype2num[at]
    else:
        return 0


def findAAindex(aa):
    if aa in ALL_AAS:
        return ALL_AAS.index(aa)
    else:
        return 0  # UNK


def read_params(
    p: str,
    as_list: bool = False,
    ignore_hisH: bool = True,
    aaname=None,
    read_mode: str = "polarH",
):
    """
    Parsing the params file

    Args:
        p: path of the params file
        as_list: if True, return list type of qs and atypes
    Return:
        atms(list), qs(dict), atypes(dict), bnds(list), repsatm(int), nchi(int)
        atms: atom list
        qs: partial charge (in this research, we used MMFF94)
        atypes: more specific atom types (e.g. Nbb, CObb etc)
        bnds: list of atom set tuples that have connection
        repsatm: NBR_ATOM index of atms
    """
    atms = []
    qs = {}
    atypes = {}
    bnds = []

    is_his = False
    repsatm = 0
    nchi = 0
    for l in open(p):
        words = l[:-1].split()
        if l.startswith("AA"):
            if "HIS" in l:
                is_his = True
        elif l.startswith("NAME"):
            aaname_read = l[:-1].split()[-1]
            if aaname is not None and aaname_read != aaname:
                return False

        if l.startswith("ATOM") and len(words) > 3:
            atm = words[1]
            atype = words[2]
            if atype[0] == "H":
                if read_mode == "heavy":
                    continue
                elif atype not in ["Hpol", "HNbb", "HO", "HS", "HN"]:
                    continue
                elif is_his and (atm in ["HE2", "HD1"]) and ignore_hisH:
                    continue

            if atype == "VIRT":
                continue
            atms.append(atm)
            atypes[atm] = atype
            qs[atm] = float(words[4])

        elif l.startswith("BOND"):
            a1, a2 = words[1:3]
            if a1 not in atms or a2 not in atms:
                continue
            border = 1
            if len(words) >= 4:
                border = {
                    "1": 1,
                    "2": 2,
                    "3": 3,
                    "CARBOXY": 2,
                    "DELOCALIZED": 2,
                    "ARO": 4,
                    "4": 4,
                    "3": 3,
                }[words[3]]

            bnds.append((a1, a2))  # ,border))

        elif l.startswith("NBR_ATOM"):
            repsatm = atms.index(l[:-1].split()[-1])
        elif l.startswith("CHI"):
            nchi += 1
        elif l.startswith("PROTON_CHI"):
            nchi -= 1

    if as_list:
        qs = [qs[atm] for atm in atms]
        atypes = [atypes[atm] for atm in atms]
    return atms, qs, atypes, bnds, repsatm, nchi


def read_pdb(
    pdb: Path,
    read_ligand: bool = False,
    aas_allowed: list = [],
    aas_disallowed: list = [],
    ignore_insertion: bool = True,
):
    """
    Parsing PDB file (read only target and ligand).

    Args:
        pdb: path of PDB file for parsing
    Return:
        resnames(list), reschains(list), xyz(dict), atms(dict)
        resnames: list of residue name (e.g. ['SER', 'ILE', ..])
        reschains: list of residue chain (e.g. [['A.1', 'A.2', ..])
        xyz: coordinate (e.g. {'A.1': {'N': [59.419, 26.851, 14.79], 'CA': [...], ...})
        atms: residue chain's atom list (e.g. {'A.1': ['N', 'CA', ...], 'A.2': [...], ...})
    """
    resnames = []
    reschains = []
    xyz = {}
    atms = {}

    for l in open(pdb):
        if not (l.startswith("ATOM") or l.startswith("HETATM")):
            continue
        atm = l[12:17].strip()
        aa3 = l[17:20].strip()

        if aas_allowed != [] and aa3 not in aas_allowed:
            continue
        if aa3 in aas_disallowed:
            continue

        reschain = l[21] + "." + l[22:27].strip()
        if ignore_insertion and l[26] != " ":
            continue

        if aa3 in AMINOACID:
            if atm == "CA":
                resnames.append(aa3)
                reschains.append(reschain)
        elif read_ligand and aa3 != "LG1":
            continue
        elif (
            read_ligand and reschain not in reschains
        ):  # "reschain not in reschains:" -> append only once
            resnames.append(aa3)  # LG1
            reschains.append(reschain)  # X.1

        if reschain not in xyz:
            xyz[reschain] = {}
            atms[reschain] = []
        xyz[reschain][atm] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
        atms[reschain].append(atm)

    return resnames, reschains, xyz, atms


def read_ligand_pdb(pdb: str, ligres: str = "LG1", read_H: bool = False):
    xyz = []
    atms = []
    for l in open(pdb):
        if not l.startswith("ATOM") and not l.startswith("HETATM"):
            continue
        atm = l[12:17].strip()
        aa3 = l[17:20].strip()
        if aa3 != ligres:
            continue
        if not read_H and atm[0] == "H":
            continue

        xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        atms.append(atm)
    xyz = np.array(xyz)
    return atms, xyz


def read_aa_sequence(pdb_file: str) -> str:
    result = []
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if PDBLineParser.atom_name(line) != "N":
                continue
            res_name = PDBLineParser.residue_name(line)
            if res_name not in AMINOACID:
                if res_name in NON_STANDARD_AA:
                    res_name = NON_STANDARD_AA[res_name]
                else:
                    continue
            res_one = AA_ONE_LETTER[res_name]
            result.append(res_one)
    return "".join(result)


def pdb_seq_align(input_pdb: str, ref_sequence: str) -> None:
    with open(input_pdb) as f:
        orig_lines = f.readlines()
    result = []
    n = 0
    for line in orig_lines:
        if not line.startswith("ATOM"):
            result.append(line)
            continue
        res_name = PDBLineParser.residue_name(line)
        if res_name not in AMINOACID:
            if res_name in NON_STANDARD_AA:
                res_name = NON_STANDARD_AA[res_name]
            else:
                continue
        res_name = AA_ONE_LETTER[res_name]
        if PDBLineParser.atom_name(line) == "N":
            n += 1
        if res_name == ref_sequence[n - 1]:
            new_line = line[:22] + str(n).rjust(4) + line[26:]
        else:
            while res_name != ref_sequence[n - 1]:
                if n == len(ref_sequence):
                    break
                n += 1
            new_line = line[:22] + str(n).rjust(4) + line[26:]
        result.append(new_line)
    return result


def get_ligand_xyz(ligand_file: str, ligand_name: str = None) -> np.ndarray:
    result = []
    # sdf file format
    if ligand_file.endswith(".sdf"):
        with open(ligand_file) as f:
            for i, line in enumerate(f):
                if i < 3:
                    continue
                if i == 3:
                    atom_num = int(line.split()[0])
                    continue
                if i >= 4 + atom_num:
                    break
                xyz = [float(t) for t in line.split()[:3]]
                result.append(xyz)
    # PDB file format
    elif ligand_file.endswith(".pdb"):
        with open(ligand_file) as f:
            for line in f:
                if not line.startswith("HETATM"):
                    continue
                if ligand_name is not None:
                    if PDBLineParser.residue_name(line) != ligand_name:
                        continue
                result.append(PDBLineParser.xyz(line))
    # mol2 file format
    elif ligand_file.endswith(".mol2"):
        with open(ligand_file) as f:
            start, check, num, i = 0, 0, 0, 0
            for line in f:
                if "@<TRIPOS>MOLECULE" in line:
                    start = 1
                    continue
                if start == 0:
                    continue
                i += 1
                if i < 2:
                    continue
                if i == 2:
                    num_atoms = int(line.split()[0])
                    continue
                if line.startswith("@<TRIPOS>ATOM"):
                    check = 1
                    continue
                if check == 0:
                    continue
                num += 1
                xyz = [float(t) for t in line.split()[2:5]]
                result.append(xyz)
                if num >= num_atoms:
                    break
    else:
        raise NotImplementedError
    return np.array(result)


def get_neighbor_chain(
    pdb_path: str,
    ligand_xyz: np.ndarray,
    center_mode: bool = True,
    dist_threshold: float = 16,
) -> list:
    selected_chains = []
    # Get protein coordinates and chain
    chain_name, rec_xyz = [], []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            rec_xyz.append(PDBLineParser.xyz(line))
            chain_name.append(PDBLineParser.chain_name(line))
    rec_xyz = np.array(rec_xyz)

    if center_mode:
        ligand_center = ligand_xyz.mean(axis=0)
        for i, xyz in enumerate(rec_xyz):
            dist = np.sqrt(np.square(xyz - ligand_center).sum())
            if dist <= dist_threshold:
                selected_chains.append(chain_name[i])
    else:
        kd = cKDTree(rec_xyz)
        indices = []
        for xyz in ligand_xyz:
            xyz = np.expand_dims(xyz, 0)
            kd_ca = cKDTree(xyz)
            indices += kd_ca.query_ball_tree(kd, dist_threshold)[0]
        indices = np.unique(indices)
        selected_chains = list(np.array(chain_name)[indices])
    return list(set(selected_chains))


def get_first_resnum(pdb: Union[str, list]) -> int:
    retval = None
    if isinstance(pdb, str):
        with open(pdb) as f:
            lines = f.readlines()
    else:
        lines = pdb
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        retval = PDBLineParser.residue_num(line)
        break
    if retval is None:
        raise ValueError("Check if there is an ATOM line in the PDB file.")
    return retval


class PDBLineParser:
    @staticmethod
    def atomid(pdb_line: str) -> int:
        """
        line[6:11]

        e.g. ATOM   4346  CD2 LEU A 377      19.620   4.797  53.831
            -> 4346
        """
        return int(pdb_line[6:11])

    @staticmethod
    def atom_name(pdb_line: str) -> str:
        """
        line[12:17]

        e.g. ATOM   4346  CD2 LEU A 377      19.620   4.797  53.831
            -> CD2
        """
        return pdb_line[12:17].strip(" ")

    @staticmethod
    def chain_name(pdb_line: str) -> str:
        """
        line[21]

        e.g. ATOM   4346  CD2 LEU A 377      19.620   4.797  53.831
            -> A
        """
        return pdb_line[21]

    @staticmethod
    def residue_num(pdb_line: str) -> int:
        """
        line[22:26]

        e.g. ATOM   4346  CD2 LEU A 377      19.620   4.797  53.831
            -> 377
        """
        return int(pdb_line[22:26].strip())

    @staticmethod
    def residue_name(pdb_line: str) -> str:
        """
        line[17:20]

        e.g. ATOM   4346  CD2 LEU A 377      19.620   4.797  53.831
            -> LEU
        """
        return pdb_line[17:20]

    @staticmethod
    def xyz(pdb_line: str) -> tuple:
        """
        line[30:38], line[38:46], line[46:54]

        e.g. ATOM   4346  CD2 LEU A 377      19.620   4.797  53.831
            -> (19.620, 4.797, 53.831)
        """
        x, y, z = float(pdb_line[30:38]), float(pdb_line[38:46]), float(pdb_line[46:54])
        return x, y, z

    @staticmethod
    def atom_type(pdb_line: str) -> str:
        """
        Last element in a line.
        """
        return pdb_line.split()[-1]

    @staticmethod
    def pdb_line_renumber(pdb_line: str, num: int) -> str:
        return pdb_line[:6] + str(num).rjust(5, " ") + pdb_line[11:]

    @staticmethod
    def new_coordinate(pdb_line: str, new_xyz: list) -> str:
        new_xyz = [round(n, 3) for n in new_xyz]
        xyz_formated = (
            str(new_xyz[0]).rjust(8)
            + str(new_xyz[1]).rjust(8)
            + str(new_xyz[2]).rjust(8)
        )
        return pdb_line[:30] + xyz_formated + pdb_line[54:]


def mol2_atom_process(ligand_mol2: str, output: str) -> None:
    with open(ligand_mol2) as f:
        orig_lines = f.readlines()
    with open(output, "w") as f:
        molecule, atom = False, False
        m_cnt, a_cnt = 0, 0
        ligand_atoms_num = {}
        for line in orig_lines:
            if (
                ("@<TRIPOS>MOLECULE" not in line)
                and ("@<TRIPOS>ATOM" not in line)
                and (molecule is False)
                and (atom is False)
            ):
                f.write(line)
                continue

            if "@<TRIPOS>MOLECULE" in line:
                molecule = True
                f.write(line)
                continue

            if molecule:
                m_cnt += 1

            if m_cnt == 1:
                f.write(line)
                continue

            if m_cnt == 2:
                atom_num = int(line.split()[0])
                m_cnt = 0
                molecule = False
                f.write(line)
                continue

            if "@<TRIPOS>ATOM" in line:
                atom = True
                f.write(line)
                continue

            if atom:
                a_cnt += 1
                if a_cnt <= atom_num:
                    ligand_atom = line.split()[1][:2]
                    if ligand_atom not in ligand_atoms_num.keys():
                        ligand_atoms_num[ligand_atom] = 0
                    else:
                        ligand_atoms_num[ligand_atom] += 1
                    atom_type_parsed = (
                        (ligand_atom + str(ligand_atoms_num[ligand_atom]))
                        .rjust(3, " ")
                        .ljust(4, " ")
                    )
                    line = line[:8] + atom_type_parsed + line[12:]
                    f.write(line)
                else:
                    atom = False
                    f.write(line)


def get_native_info(
    xyz_r, xyz_l, bnds_l=[], atms_l=[], contact_dist=5.0, shift_nl=True
):
    nr = len(xyz_r)
    nl = len(xyz_l)

    # get list of ligand bond connectivity
    if bnds_l != []:
        bnds_l = [(i, j) for i, j in bnds_l]
        angs_l = []
        for i, b1 in enumerate(bnds_l[:-1]):
            for b2 in bnds_l[i + 1 :]:
                if b1[0] == b2[0]:
                    angs_l.append((b1[1], b2[1]))
                elif b1[0] == b2[0]:
                    angs_l.append((b1[1], b2[1]))
                elif b1[1] == b2[1]:
                    angs_l.append((b1[0], b2[0]))
                elif b1[0] == b2[1]:
                    angs_l.append((b1[1], b2[0]))
                elif b1[1] == b2[0]:
                    angs_l.append((b1[0], b2[1]))
        bnds_l += angs_l

    dmap = np.array(
        [
            [np.dot(xyz_l[i] - xyz_r[j], xyz_l[i] - xyz_r[j]) for j in range(nr)]
            for i in range(nl)
        ]
    )
    dmap = np.sqrt(dmap)
    contacts = np.where(dmap < contact_dist)
    if shift_nl:
        contacts = [(j, contacts[1][i] + nl) for i, j in enumerate(contacts[0])]
        dco = [dmap[i, j - nl] for i, j in contacts]
    else:
        contacts = [(j, contacts[1][i]) for i, j in enumerate(contacts[0])]
        dco = [dmap[i, j] for i, j in contacts]

    dmap_l = np.array(
        [
            [
                np.sqrt(np.dot(xyz_l[i] - xyz_l[j], xyz_l[i] - xyz_l[j]))
                for j in range(nl)
            ]
            for i in range(nl)
        ]
    )
    contacts_l = np.where(dmap_l < contact_dist)
    contacts_l = [
        (j, contacts_l[1][i])
        for i, j in enumerate(contacts_l[0])
        if j < contacts_l[1][i] and ((j, contacts_l[1][i]) not in bnds_l)
    ]

    dco += [dmap_l[i, j] for i, j in contacts_l]
    contacts += contacts_l

    return contacts, dco


def distogram(
    distance_map: torch.Tensor,
    min_length: int,
    max_length: int,
    num_classes: int,
    abs_min_diff: int,
) -> np.ndarray:
    device = distance_map.device
    distance_map = torch.clamp(distance_map, min_length, max_length)
    bins = torch.linspace(min_length, max_length, num_classes).to(device)
    bins = torch.where(abs(bins) < abs_min_diff, torch.tensor([0.0]).to(device), bins)
    binned_distance_map = torch.bucketize(distance_map, bins)
    return binned_distance_map


def fa2gentype(fats):
    """
    Mapping atypes to "gentype2num"

    Parameters:
        fats (iterable): atypes list
    """
    gts = {
        "Nbb": "Nad",
        "Npro": "Nad3",
        "NH2O": "Nad",
        "Ntrp": "Nin",
        "Nhis": "Nim",
        "NtrR": "Ngu2",
        "Narg": "Ngu1",
        "Nlys": "Nam",
        "CAbb": "CS1",
        "CObb": "CDp",
        "CH1": "CS1",
        "CH2": "CS2",
        "CH3": "CS3",
        "COO": "CDp",
        "CH0": "CR",
        "aroC": "CR",
        "CNH2": "CDp",
        "OCbb": "Oad",
        "OOC": "Oat",
        "OH": "Ohx",
        "ONH2": "Oad",
        "S": "Ssl",
        "SH1": "Sth",
        "HNbb": "HN",
        "HS": "HS",
        "Hpol": "HO",
        "Phos": "PG5",
        "Oet2": "OG3",
        "Oet3": "OG3",  # Nucleic acids
    }

    gents = []
    # if element not in gentype2num, then mapping to gentype2num using gts
    for at in fats:
        if at in gentype2num:
            gents.append(at)
        else:
            gents.append(gts[at])
    return gents


def defaultparams(
    aa, datapath="/home/bbh9955/programs/Rosetta/residue_types", extrapath=""
):
    """
    Get params path of aa

    Args:
        aa: element for getting params
    Return:
        path of params file
    """
    # First search through Rosetta database
    if aa in AMINOACID:
        p = "%s/l-caa/%s.params" % (datapath, aa)
        return p

    p = "%s/%s.params" % (extrapath, aa)
    if not os.path.exists(p):
        p = "%s/LG.params" % (extrapath)
    if not os.path.exists(p):
        sys.exit(
            "Failed to found relevant params file for aa:"
            + aa
            + ",  -- check if LG.params exits"
        )
    return p


def get_AAtype_properties(ignore_hisH=True, extrapath="", extrainfo={}):
    """
    Get properties of atypes

    Return: qs_aa(dict), atypes_aa(dict), atms_aa(dict), bnds_aa(dict), repsatm_aa(dict)
        each dictionary has 32 number keys (AMINOACID+NUCLEICACID+METAL) with starting number 1.
        "0" key means "UNK".
        atypes_aa[i]->dict: dictionary of more specific atom types (e.g. Nbb, CObb etc) (atom: atype)
        atms_aa[i]->list: atom list
        bnds_aa[i]->list: list of atom set tuples that have connection
        repsatm_aa[i]->int: index of representative atom
    """
    qs_aa = {}
    atypes_aa = {}
    atms_aa = {}
    bnds_aa = {}
    repsatm_aa = {}

    iaa = 0  # "UNK"
    for aa in AMINOACID:
        iaa += 1
        p = defaultparams(aa)
        atms, q, atypes, bnds, repsatm, _ = read_params(p)
        atypes_aa[iaa] = fa2gentype([atypes[atm] for atm in atms])
        qs_aa[iaa] = q
        atms_aa[iaa] = atms
        bnds_aa[iaa] = bnds
        if aa in AMINOACID:
            repsatm_aa[iaa] = atms.index("CA")
        else:
            repsatm_aa[iaa] = repsatm

    if extrapath != "":
        params = glob.glob(extrapath + "/*params")
        for p in params:
            aaname = p.split("/")[-1].replace(".params", "")
            args = read_params(p, aaname=aaname)
            if not args:
                print("Failed to read extra params %s, ignore." % p)
                continue
            else:
                # print("Read %s for the extra res params for %s"%(p,aaname))
                pass
            atms, q, atypes, bnds, repsatm = args
            atypes = [atypes[atm] for atm in atms]  # same atypes
            extrainfo[aaname] = (q, atypes, atms, bnds, repsatm)
    if extrainfo != {}:
        print("Extra residues read from %s: " % extrapath, list(extrainfo.keys()))
    return qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa


def read_sasa(f, reschains):
    read_cont = False
    cbcount = {}
    sasa = {}

    for l in open(f):
        if l.startswith("   1"):
            read_cont = True
            continue
        if not read_cont:
            continue
        #    1 A    1   PRO    0.743   77.992  100.816  17
        chain = l[5]
        resno = l[7:12].strip()
        reschain = "%s.%s" % (chain, resno)

        words = l[12:-1].split()
        rsa_sc = float(words[1])
        asa_sc = float(words[2])
        asa_tot = float(words[3])
        ncb = int(words[4])

        sasa[reschain] = min(1.0, asa_tot / 200.0)
        cbcount[reschain] = min(1.0, ncb / 50.0)

    # assign neutral if missing
    for res in reschains:
        if res not in sasa:
            print("Missing res!", res, "; assign neutral value")
            sasa[res] = 0.25
            cbcount[res] = 0.5
    return cbcount, sasa


def cosine_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    t = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(t)


def needleman_wunsch(
    seq1: str, seq2: str, match: int = 1, mismatch: int = -1, gap: int = -1
) -> Tuple[int, str, str]:
    """
    Perform pairwise sequence alignment using the Needleman-Wunsch algorithm.

    Args:
        - seq1 (str): The first amino acid sequence.
        - seq2 (str): The second amino acid sequence.
        - match (int): Score for a match.
        - mismatch (int): Score for a mismatch.
        - gap (int): Score for a gap.

    Returns:
        - alignment_score (int): The alignment score.
        - aligned_seq1 (str): The aligned version of seq1.
        - aligned_seq2 (str): The aligned version of seq2.
    """

    # Initialize score matrix
    n = len(seq1)
    m = len(seq2)
    score_matrix = np.zeros((m + 1, n + 1))

    # Initialize traceback matrix
    traceback_matrix = np.zeros((m + 1, n + 1))

    # Initialize first row and column of the score matrix
    for i in range(1, n + 1):
        score_matrix[0][i] = gap * i
    for j in range(1, m + 1):
        score_matrix[j][0] = gap * j

    # Fill in the score matrix and the traceback matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_mismatch_score = match if seq1[j - 1] == seq2[i - 1] else mismatch
            diag_score = score_matrix[i - 1][j - 1] + match_mismatch_score
            up_score = score_matrix[i - 1][j] + gap
            left_score = score_matrix[i][j - 1] + gap
            max_score = max(diag_score, up_score, left_score)
            score_matrix[i][j] = max_score
            if max_score == diag_score:
                traceback_matrix[i][j] = 1  # Diagonal traceback
            elif max_score == up_score:
                traceback_matrix[i][j] = 2  # Upwards traceback
            else:
                traceback_matrix[i][j] = 3  # Leftwards traceback

    # Traceback to construct aligned sequences
    aligned_seq1 = ""
    aligned_seq2 = ""
    i, j = m, n
    while i > 0 or j > 0:
        if traceback_matrix[i][j] == 1:
            aligned_seq1 = seq1[j - 1] + aligned_seq1
            aligned_seq2 = seq2[i - 1] + aligned_seq2
            i -= 1
            j -= 1
        elif traceback_matrix[i][j] == 2:
            aligned_seq1 = "-" + aligned_seq1
            aligned_seq2 = seq2[i - 1] + aligned_seq2
            i -= 1
        else:
            aligned_seq1 = seq1[j - 1] + aligned_seq1
            aligned_seq2 = "-" + aligned_seq2
            j -= 1

    alignment_score = int(score_matrix[m][n])

    return alignment_score, aligned_seq1, aligned_seq2


def get_mw(atoms_list: list) -> float:
    mw_list = []
    for atom in atoms_list:
        mw_list.append(float(AtomicWeightsDecimal[atom.capitalize()]["standard"]))

    return sum(mw_list)
