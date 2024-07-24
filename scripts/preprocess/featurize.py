import sys
import os
import numpy as np
from glob import glob
from scipy.spatial import distance_matrix
import argparse
import copy
from pathlib import Path
from typing import Tuple, List, Optional

from denoiser.utils.chem_utils import (
    ALL_AAS,
    defaultparams,
    read_params,
    get_AAtype_properties,
    read_pdb,
    findAAindex,
    get_native_info,
)


# Input: Protein-ligand complexes (decoys)
# -> It needs to be processed like ligand residue name (LG1)
# Outputs: prop.npz, lig.npz, plddt.npy, occlusion.pkl, LG.params


def get_ligpdbs(
    input_path: Path,
    exclude: list = ["holo.pdb", "pocket.pdb", "holo_orig.pdb", "native.pdb"],
) -> List[Path]:
    """
    Args:
        input_path: path that docked files exist
    Return:
        path list of docked files in input_path
    """
    result = input_path.glob("*.pdb")
    result = [t for t in result if t.name not in exclude]
    return result


def read_ligand_params(xyz: dict, aas: list, chainres: list, extrapath: str = ""):
    """
    Returns:
        atms_lig, q_lig, atypes_lig, bnds_lig, repsatm_lig, atmres_lig
    """
    atms_lig = []
    q_lig = []
    atypes_lig = []
    bnds_lig = []
    repsatm_lig = []
    atmres_lig = []

    natm = 0
    for ires, aa in enumerate(aas):
        rc = chainres[ires]
        p = defaultparams(aa, extrapath=extrapath)
        atms_aa, qs_aa, atypes_aa, bnds_aa, repsatm_aa, nchi_aa = read_params(
            p, as_list=True
        )

        # Make sure all atom exists
        atms_aa_lig = []
        for atm, q, atype in zip(atms_aa, qs_aa, atypes_aa):
            if atm not in xyz[rc]:
                continue
            atms_aa_lig.append(atm)
            q_lig.append(q)
            atypes_lig.append(atype)
            atmres_lig.append((rc, atm))
        atms_lig += atms_aa_lig

        repsatm_lig.append(repsatm_aa)
        for a1, a2 in bnds_aa:
            if (a1 not in atms_aa_lig) or (a2 not in atms_aa_lig):
                continue
            bnds_lig.append(
                (atms_aa_lig.index(a1) + natm, atms_aa_lig.index(a2) + natm)
            )

        natm += len(atms_lig)

    bnds_lig = np.array(bnds_lig, dtype=int)

    return atms_lig, q_lig, atypes_lig, bnds_lig, repsatm_lig, atmres_lig


def sasa_from_xyz(
    xyz: np.ndarray, reschains: list, atmres_rec: List[Tuple[str, str]]
) -> List[float]:
    """
    Compute sasa from xyz coordinates

    Args:
        xyz: repatm's coordinate
    """
    D = distance_matrix(xyz, xyz)
    cbcounts = np.sum(D < 12.0, axis=0) - 1.0

    # Convert to apprx sasa
    cbnorm = cbcounts / 50.0
    sasa_byres = 1.0 - cbnorm ** (2.0 / 3.0)
    sasa_byres = np.clip(sasa_byres, 0.0, 1.0)

    # By atom
    sasa = [sasa_byres[reschains.index(res)] for res, atm in atmres_rec]

    return sasa


def per_atm_lddt(
    xyz_lig: np.ndarray, xyz_rec: np.ndarray, dco: list, contact: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute fnat and lddt

    Args:
        xyz_lig: coordinates of the ligand
        xyz_rec: coordinates of the receptor(protein)
        dco: distance between atom of the ligand and protein in native complex
    Return:
        fnat, lddt_per_atm
    """
    xyz = np.concatenate([xyz_lig, xyz_rec])
    nco = len(dco)
    natm = len(xyz_lig)
    deltad = np.zeros(nco)
    deltad_per_atm = [[] for i in range(natm)]

    for i, (a1, a2) in enumerate(contact):  # a1,a2 are lig, rec atmidx
        dv = xyz[a1] - xyz[a2]
        d = np.sqrt(np.dot(dv, dv))
        deltad[i] = abs(dco[i] - d)
        deltad_per_atm[a1].append(deltad[i])

    fnat = (
        np.sum(deltad < 1.0)
        + np.sum(deltad < 2.0)
        + np.sum(deltad < 3.0)
        + np.sum(deltad < 4.0)
    )
    fnat /= 4.0 * (nco + 0.001)

    lddt_per_atm = np.zeros(natm)
    for i, col in enumerate(deltad_per_atm):
        col = np.array(col)
        lddt_per_atm[i] = (
            np.sum(col < 0.5)
            + np.sum(col < 1.0)
            + np.sum(col < 2.0)
            + np.sum(col < 4.0)
        )
        lddt_per_atm[i] /= (len(col) + 0.001) * 4.0
    return fnat, lddt_per_atm


def featurize_receptor(
    input_path: Path, outf: Path, pdb: str, extrapath: str = "", store_npz: bool = True
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Extract features from the target and save to *.prop.npz.

    Components of *.prop.npz:
        aas_rec, xyz_rec, atypes_rec, charge_rec, bnds_rec, cbcounts_rec, sasa_rec, residue_idx,
        repsatm_idx, reschains, atmnames, resnames,

    Args:
        outf: path to save prop.npz
        store_npz (bool): if True, save prop.npz file
    Return:
        xyz_rec: coordinate of pdb (shape: (atom_number,3))
        atmres_rec: list of (residue chain, atom) sets (e.g. [('A.1', 'N'), ('A.1', 'CA'), ...])
    """

    extra = {}
    qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa = get_AAtype_properties(
        extrapath=extrapath, extrainfo=extra
    )
    # Parsing PDB file
    resnames, reschains, xyz, _ = read_pdb(
        input_path / pdb, read_ligand=True, aas_disallowed=["LG1"]
    )

    # Atom-wise features
    q_rec = []
    atypes_rec = []
    xyz_rec = []
    atmres_rec = []
    aas_rec = []  # residue index in ALL_AAS
    residue_idx = []
    # Residue-wise features
    bnds_rec = []
    repsatm_idx = []
    iaas = []  # residue index in ALL_AAS

    for i, (resname, reschain) in enumerate(zip(resnames, reschains)):
        if resname in extra:  # UNK
            iaa = 0
            qs, atypes, atms, bnds_, repsatm = extra[resname]
        elif resname in ALL_AAS:
            iaa = findAAindex(resname)
            qs, atypes, atms, bnds_, repsatm = (
                qs_aa[iaa],
                atypes_aa[iaa],
                atms_aa[iaa],
                bnds_aa[iaa],
                repsatm_aa[iaa],
            )
        else:
            print("unknown residue: %s, skip" % resname)
            continue

        natm = len(xyz_rec)
        atms_r = []
        iaas.append(iaa)
        for iatm, atm in enumerate(atms):
            is_repsatm = iatm == repsatm

            if atm not in xyz[reschain]:
                if is_repsatm:
                    return False
                continue
            # It should be checked
            if resname == "CYS" and atm == "HG":
                continue

            atms_r.append(atm)
            q_rec.append(qs[atm])
            atypes_rec.append(atypes[iatm])
            aas_rec.append(iaa)
            xyz_rec.append(xyz[reschain][atm])
            atmres_rec.append((reschain, atm))
            residue_idx.append(i)
            if is_repsatm:
                repsatm_idx.append(natm + iatm)

        bnds = [
            [atms_r.index(atm1), atms_r.index(atm2)]
            for atm1, atm2 in bnds_
            if atm1 in atms_r and atm2 in atms_r
        ]

        # Make sure all bonds are right
        for i1, i2 in copy.copy(bnds):
            dv = np.array(xyz_rec[i1 + natm]) - np.array(xyz_rec[i2 + natm])
            d = np.sqrt(np.dot(dv, dv))
            if d > 2.0:
                print(
                    "Warning, abnormal bond distance: ",
                    input_path,
                    resname,
                    reschain,
                    i1,
                    i2,
                    atms_r[i1],
                    atms_r[i2],
                    d,
                )

        bnds = np.array(bnds, dtype=int)

        if i == 0:
            bnds_rec = bnds
        elif bnds_ != []:
            bnds += natm
            bnds_rec = np.concatenate([bnds_rec, bnds])

    xyz_rec = np.array(xyz_rec)

    sasa = sasa_from_xyz(xyz_rec[repsatm_idx], reschains, atmres_rec)
    if store_npz:
        np.savez(
            outf,
            # Atom-wise features
            aas_rec=aas_rec,
            xyz_rec=xyz_rec,  # just native info
            atypes_rec=atypes_rec,
            charge_rec=q_rec,
            bnds_rec=bnds_rec,
            sasa_rec=sasa,
            residue_idx=residue_idx,
            # Residue-wise features (only for receptor)
            repsatm_idx=repsatm_idx,
            reschains=reschains,
        )

    return xyz_rec, atmres_rec


def featurize_complexes(
    input_path: Path,
    output_path: Optional[Path] = None,
    outprefix: Optional[str] = None,
    paramspath: Optional[str] = None,
    native_structure: Optional[str] = None,
    exclude: Optional[str] = None,
    cross_docking: bool = False,
) -> None:
    """
    Generate input files for the model inference. (lig.npz, prop.npz)

    Args:
        paramspath: path of LG.params
        native_structure: File name of the native structure with PDB format in the input_path
    """
    input_path = input_path.absolute()
    if output_path is None:
        output_path = input_path
    output_path = output_path.absolute()
    if paramspath is None:
        paramspath = input_path
    if outprefix is None:
        outprefix = input_path.name
    if exclude is not None:
        with open(os.path.join(input_path, exclude)) as f:
            exclude_list = [t.strip() for t in f.readlines()]
            exclude_list += ["holo.pdb", "pocket.pdb", "native.pdb"]
    else:
        exclude_list = ["holo.pdb", "pocket.pdb", "native.pdb"]
    if cross_docking:
        assert (
            native_structure is not None
        ), "cross_docking option is for calculating the complex lDDT. Therefore, It requires the native structure."

    # Get list of files that would be re-scored
    ligpdbs = get_ligpdbs(input_path, exclude_list)

    if native_structure is not None:
        pdb = native_structure
        ligpdbs = [t for t in ligpdbs if native_structure not in t.name]
        training = True
    else:
        pdb = ligpdbs[0].name
        training = False

    if cross_docking:
        rep_xyz_rec0, atmres_rec = featurize_receptor(
            input_path,
            "%s/%s.prop.npz" % (output_path, outprefix),
            pdb=ligpdbs[0].split("/")[-1],
        )  # decoy
        xyz_rec0, native_atmres_rec = featurize_receptor(
            input_path,
            None,
            pdb=pdb,  # native
            store_npz=False,
        )
        t_xyz_rec0, t_atmres_rec = [], []
        for i in range(len(native_atmres_rec)):
            if native_atmres_rec[i] not in atmres_rec:
                continue
            t_xyz_rec0.append(list(xyz_rec0[i]))
            t_atmres_rec.append(native_atmres_rec[i])
        xyz_rec0 = np.array(t_xyz_rec0)
        native_atmres_rec = t_atmres_rec
    else:
        xyz_rec0, atmres_rec = featurize_receptor(
            input_path, "%s/%s.prop.npz" % (output_path, outprefix), pdb=pdb
        )  # native or decoy
        native_atmres_rec = atmres_rec

    _aas, _reschains, _xyz, _ = read_pdb(
        os.path.join(input_path, pdb), read_ligand=True
    )
    ligchain = _reschains[-1].split(".")[0]  # Take the last chain as the ligand chain

    if training:
        _reschains_lig = [a for i, a in enumerate(_reschains) if a[0] == ligchain]
        _aas_lig = [_aas[i] for i, rc in enumerate(_reschains) if rc in _reschains_lig]
        args = read_ligand_params(_xyz, _aas_lig, _reschains_lig, extrapath=paramspath)
        _ligatms, _, _, _bnds_lig, _, native_atmres_lig = args
        xyz_lig = np.array([_xyz[res][atm] for res, atm in native_atmres_lig])
        contacts, dco = get_native_info(xyz_rec0, xyz_lig, _bnds_lig, _ligatms)

    xyz_lig, xyz_rec = [], []
    lddt, fnat = [], []
    pnames = []

    nfail = 0
    read_first = True
    for pdb in ligpdbs:
        pname = pdb.name[:-4]

        try:
            _aas, _reschains, _xyz, _ = read_pdb(pdb, read_ligand=True)

            ligchain = _reschains[-1].split(".")[
                0
            ]  # Take the last chain as the ligand chain -> 'X'
            _reschains_lig = [
                rc for i, rc in enumerate(_reschains) if rc[0] == ligchain
            ]
            _aas_lig = [
                _aas[i] for i, rc in enumerate(_reschains) if rc in _reschains_lig
            ]

            args = read_ligand_params(
                _xyz, _aas_lig, _reschains_lig, extrapath=paramspath
            )
            if read_first:
                _ligatms, _q_lig, _atypes_lig, _bnds_lig, _repsatm_lig, atmres_lig = (
                    args
                )
                _aas_ligA = [
                    findAAindex(_aas[_reschains.index(res)]) for res, atm in atmres_lig
                ]
            else:
                _, _, _, _, _, atmres_lig = args
            # Receptor atom coordinates
            if training:
                _fnat_xyz_rec = []
                for res, atm in native_atmres_rec:
                    if res not in _xyz.keys():
                        continue
                    if atm not in _xyz[res].keys():
                        continue
                    _fnat_xyz_rec.append(_xyz[res][atm])
                _fnat_xyz_rec = np.array(_fnat_xyz_rec)
            _xyz_rec = np.array(
                [_xyz[res][atm] for res, atm in atmres_rec if res in _xyz.keys()]
            )
            # Ligand atom coordinates
            if training:
                _fnat_xyz_lig = []
                for res, atm in native_atmres_lig:
                    if res not in _xyz.keys():
                        continue
                    if atm not in _xyz[res].keys():
                        continue
                    _fnat_xyz_lig.append(_xyz[res][atm])
                _fnat_xyz_lig = np.array(_fnat_xyz_lig)
            _xyz_lig = np.array([_xyz[res][atm] for res, atm in atmres_lig])
            read_first = False

        except:
            print("Error occured while reading %s: skip." % pdb)
            nfail += 1
            continue

        if len(_ligatms) != len(_xyz_lig):
            sys.exit(
                "Different length b/w ref and decoy ligand atms! %d vs %d in %s"
                % (len(_ligatms), len(_xyz_lig), pdb)
            )

        # In training, calculate target labels
        _fnat, lddt_per_atm = 0.0, 0.0
        if training:
            _fnat, lddt_per_atm = per_atm_lddt(
                _fnat_xyz_lig, _fnat_xyz_rec, dco, contacts
            )

        xyz_rec.append(_xyz_rec)
        xyz_lig.append(_xyz_lig)
        lddt.append(lddt_per_atm)
        fnat.append(_fnat)
        pnames.append(pname)

    aas_lig = [_aas_ligA]
    bnds_lig = [_bnds_lig]
    q_lig = [_q_lig]
    atypes_lig = [_atypes_lig]
    repsatm_lig = [_repsatm_lig]

    if nfail > 0.5 * len(ligpdbs):
        print("too many failed... return none for %s" % input_path)
        return

    np.savez(
        "%s/%s.lig.npz" % (output_path, outprefix),
        aas_lig=aas_lig,
        xyz=xyz_lig,
        xyz_rec=xyz_rec,
        atypes_lig=atypes_lig,
        charge_lig=q_lig,
        bnds_lig=bnds_lig,
        repsatm_lig=repsatm_lig,
        lddt=lddt,
        fnat=fnat,
        name=pnames,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Target directory containing only decoys to be re-scored"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path for saving files that have extracted features",
    )
    parser.add_argument(
        "--native",
        default=None,
        help="(Optinal) File name of native structures for calculting target labels. It should be in the same directory with inputs",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="(Optinal) One text file name, which is in input directory, that contains file names to be not included to extract features.",
    )
    parser.add_argument(
        "--cross-docking",
        action="store_true",
        help="(Optinal) Option for calculating complex lDDT for the cross-docking task. It is calculated taking into account differences in amino acid sequence.",
    )
    args = parser.parse_args()
    out_path = Path(args.output) if args.output is not None else args.output

    featurize_complexes(
        input_path=Path(args.input),
        output_path=out_path,
        native_structure=args.native,
        exclude=args.exclude,
        cross_docking=args.cross_docking,
    )


if __name__ == "__main__":
    main()
