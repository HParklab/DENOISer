import os
from os.path import join
from pathlib import Path
import sys
import argparse
import shutil
from pathlib import Path
import re
from typing import Optional, List

from denoiser.utils.train_utils import files_only_pdb
from denoiser.utils.chem_utils import (
    get_neighbor_chain,
    get_ligand_xyz,
    PDBLineParser as PDBLP,
    read_aa_sequence,
    needleman_wunsch,
    pdb_seq_align,
)
from denoiser.preprocess import LigandParams, ExtractAFpLDDT

# Outputs: holo.pdb, plddt.npy, LG.params


class Preprocess:
    def __init__(
        self,
        input_path: Path,
        prefix: str,
        output_path: Optional[Path] = None,
        af_plddt_save_path: Optional[Path] = None,
        af_model: Optional[str] = None,
        ligand_name: str = "LG1",
        ligand_mol2: Optional[str] = None,
        complex_to_extract_ligand: Optional[str] = None,
        add_h: bool = False,
        native: Optional[str] = None,
        exclude: Optional[str] = None,
        skip_ligand_processing: bool = False,
        cross_docking: bool = False,
    ):
        """
        Args:
            save_path: Path for saving pLDDT
        """
        assert not (
            ligand_mol2 is not None and complex_to_extract_ligand is not None
        ), "Complex PDB files are not required if the ligand mol2 files are given."

        self.input_path = input_path
        if output_path is None:
            output_path = input_path
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.ligand_name = ligand_name
        self.af_model = af_model
        self.plddt_extract = ExtractAFpLDDT(
            input_path, prefix, af_plddt_save_path, af_model
        )
        if str(self.input_path) == str(self.output_path):
            original_pdbs_dir = "./original_pdbs"
            self.ligand_param = LigandParams(
                input_path,
                output_path,
                ligand_mol2,
                original_pdbs_dir,
                ligand_name=ligand_name,
                add_h=add_h,
            )
            self.original_pdbs_dir = original_pdbs_dir
        else:
            self.ligand_param = LigandParams(
                input_path, output_path, ligand_mol2, add_h=add_h
            )
        self.native = self.input_path / native if native is not None else None

        self.exclude_list = None
        if exclude is not None:
            with open(join(input_path, exclude)) as f:
                self.exclude_list = [t.strip() for t in f.readlines()]

        self.skip_ligand_processing = skip_ligand_processing
        self.cross_docking = cross_docking

        self.complex_to_extract_ligand = complex_to_extract_ligand

    def run(self) -> None:
        # Remain protein chains only in 15A from the center of ligand
        self.filter_chain()

        # Concatenate multi-chains & change ligand name
        self.concat_chains_change_ligand()

        # Optional. Sequence matching for calculating complex lDDT
        if self.native:
            self.sequence_align()

        # Remove duplicated residues
        self.remove_duplicate()

        # Generate holo.pdb
        if not self.skip_ligand_processing:
            self.save_holo()

        # Save pLDDT (if it were not model docking, save array with ones)
        self.plddt_extract.save_plddt()

        # Save ligand parameters (LG.params)
        if not self.skip_ligand_processing:
            self.ligand_param.save_params(self.complex_to_extract_ligand)

    def filter_chain(self) -> None:
        """
        Filtering chains in the target protein for multi-chain proteins.

        Returns:
            write file: save chain filtered PDB files to 'output_path'
        """
        if self.input_path == self.output_path:
            os.makedirs(self.input_path.joinpath(self.original_pdbs_dir), exist_ok=True)
        all_pdbs = self.all_pdbs
        rep_pdb = self.input_path.joinpath(all_pdbs[0])

        ligand_xyz = get_ligand_xyz(str(rep_pdb), self.ligand_name)

        # Select chains to be remained
        selected_chains = get_neighbor_chain(rep_pdb, ligand_xyz, False, 4)
        if len(selected_chains) < 1:
            print("Verify contact between ligand and receptor or Input ligand name.")
            sys.exit()

        # Filtering
        for pdb in all_pdbs:
            if self.input_path == self.output_path:
                each_pdb_dir = self.input_path.joinpath(self.original_pdbs_dir, pdb)
                if not os.path.exists(each_pdb_dir):
                    shutil.move(self.input_path.joinpath(pdb), each_pdb_dir)
            else:
                each_pdb_dir = self.input_path.joinpath(pdb)

            with open(join(self.output_path, pdb), "w") as f1:
                with open(each_pdb_dir) as f2:
                    for line in f2:
                        if not line.startswith("ATOM"):
                            f1.write(line)
                            continue
                        if (
                            self.cross_docking
                            and self.native
                            and pdb == os.path.basename(self.native)
                        ):
                            f1.write(line)
                            continue
                        if PDBLP.chain_name(line) in selected_chains:
                            f1.write(line)

        if self.native is not None:
            shutil.copy(
                self.input_path.joinpath(self.native.name),
                self.output_path.joinpath(self.native.name),
            )

    def concat_chains_change_ligand(self) -> None:
        all_pdbs = self.all_pdbs
        if self.native is not None:
            all_pdbs += [self.native.name]
        for pdb in all_pdbs:
            each_pdb_dir = self.output_path.joinpath(pdb)
            with open(each_pdb_dir) as f:
                orig_lines = f.readlines()
            with open(each_pdb_dir, "w") as f:
                ligand_atoms_num = {}
                for line in orig_lines:
                    if line.startswith("TER"):
                        continue
                    # Change ligand lines
                    if line.startswith("HETATM"):
                        if PDBLP.residue_name(line) != self.ligand_name:
                            continue
                        else:
                            line = line[:17] + "LG1" + line[20:]
                            line = line[:21] + "X   1" + line[26:]
                            # For duplicated atom types in the ligand
                            if not self.skip_ligand_processing:
                                # ligand_atom = PDBLP.atom_name(line)[:2]
                                ligand_atom = re.sub(
                                    r"[^a-zA-Z]", "", PDBLP.atom_name(line)
                                )
                                if ligand_atom not in ligand_atoms_num.keys():
                                    ligand_atoms_num[ligand_atom] = 0
                                else:
                                    ligand_atoms_num[ligand_atom] += 1
                                atom_type_parsed = (
                                    ligand_atom + str(ligand_atoms_num[ligand_atom])
                                ).ljust(5, " ")
                                line = line[:12] + atom_type_parsed + line[17:]
                            f.write(line)
                            continue
                    # Concat chains in protein
                    if not line.startswith("ATOM"):
                        f.write(line)
                        continue
                    # Without renumbering (renumbering is required for calculating lDDT)
                    new_line = line[:21] + "A" + line[22:]
                    f.write(new_line)

    def sequence_align(self) -> None:
        native = join(self.output_path, os.path.basename(self.native))
        all_pdbs = self.all_pdbs
        rep_pdb = join(self.output_path, all_pdbs[0])
        # Get aligned sequences
        ref_seq = read_aa_sequence(native)
        tar_seq = read_aa_sequence(rep_pdb)
        if len(ref_seq) >= len(tar_seq):
            _, ref_seq, tar_seq = needleman_wunsch(ref_seq, tar_seq)
        else:
            _, tar_seq, ref_seq = needleman_wunsch(tar_seq, ref_seq)
        # Rep decoy
        chain_num_atom: dict[str, List[str]] = {}
        with open(rep_pdb) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                chain_num = PDBLP.chain_name(line) + "." + str(PDBLP.residue_num(line))
                if chain_num not in chain_num_atom:
                    chain_num_atom[chain_num] = []
                chain_num_atom[chain_num].append(PDBLP.atom_name(line))
        # Native
        new_lines = pdb_seq_align(native, ref_seq)
        with open(native, "w") as f:
            new_protein_lines = []
            for line in new_lines:
                if not line.startswith("ATOM"):
                    continue
                chain_num = PDBLP.chain_name(line) + "." + str(PDBLP.residue_num(line))
                if chain_num not in chain_num_atom:
                    continue
                if PDBLP.atom_name(line) in chain_num_atom[chain_num]:
                    new_protein_lines.append(line)
            # new_protein_lines = [l for l in new_lines if l.startswith("ATOM") and
            #                      PDBLP.atom_name(l) in chain_num_atom[PDBLP.chain_name(l) + "." + str(PDBLP.residue_num(l))]]
            f.writelines(new_protein_lines)
            new_ligand_lines = [l for l in new_lines if l.startswith("HETATM")]
            f.writelines(new_ligand_lines)
        # Decoys
        for d in all_pdbs:
            d = join(self.output_path, d)
            new_lines = pdb_seq_align(d, tar_seq)
            with open(d, "w") as f:
                new_protein_lines = [l for l in new_lines if l.startswith("ATOM")]
                f.writelines(new_protein_lines)
                new_ligand_lines = [l for l in new_lines if l.startswith("HETATM")]
                f.writelines(new_ligand_lines)

    def remove_duplicate(self) -> None:
        all_pdbs = self.all_pdbs
        for pdb in all_pdbs:
            resnums = []
            skip_num = -10000
            diff = 0
            each_pdb_dir = join(self.output_path, pdb)
            with open(each_pdb_dir) as f:
                orig_lines = f.readlines()
            with open(each_pdb_dir, "w") as f:
                for line in orig_lines:
                    if not line.startswith("ATOM"):
                        f.write(line)
                        continue
                    # Remove iCode character & renumbering
                    icode = line[26]
                    if PDBLP.atom_name(line) == "N" and icode != " ":
                        if PDBLP.residue_num(line) in resnums:
                            diff += 1
                    num = PDBLP.residue_num(line) + diff
                    line = line[:22] + str(num).rjust(4, " ") + " " + line[27:]
                    # For skip duplicated residues
                    resnum = PDBLP.residue_num(line)
                    if resnum == skip_num:
                        continue
                    # If there are duplicate residues, only keep the one first seen
                    if PDBLP.atom_name(line) == "N" and resnum in resnums:
                        skip_num = resnum
                        continue
                    f.write(line)
                    if PDBLP.atom_name(line) == "N":
                        resnums.append(resnum)

    def save_holo(self, ligand_name: str = "LG1") -> None:
        save_path = join(self.output_path, "holo.pdb")
        with open(save_path, "w") as f1:
            if self.native is None:
                with open(join(self.output_path, self.repcom)) as f2:
                    atom_num = 1
                    for line in f2:
                        if not line.startswith("ATOM") and not line.startswith(
                            "HETATM"
                        ):
                            continue
                        if line.startswith("ATOM") and PDBLP.atom_type(line) == "H":
                            continue
                        if line.startswith("HETATM"):
                            if PDBLP.residue_name(line) != ligand_name:
                                continue
                            line = self.hetatm_parsing(line)
                        f1.write(PDBLP.pdb_line_renumber(line, atom_num))
                        atom_num += 1
            else:
                atom_num = 1
                # Receptor
                with open(join(self.output_path, self.repcom)) as f2:
                    for line in f2:
                        if not line.startswith("ATOM") and not line.startswith(
                            "HETATM"
                        ):
                            continue
                        if line.startswith("ATOM") and PDBLP.atom_type(line) == "H":
                            continue
                        if line.startswith("HETATM"):
                            continue
                        f1.write(PDBLP.pdb_line_renumber(line, atom_num))
                        atom_num += 1
                # Ligand
                with open(join(self.output_path, os.path.basename(self.native))) as f2:
                    for line in f2:
                        if not line.startswith("HETATM"):
                            continue
                        if PDBLP.residue_name(line) != ligand_name:
                            continue
                        line = self.hetatm_parsing(line)
                        f1.write(PDBLP.pdb_line_renumber(line, atom_num))
                        atom_num += 1

    @staticmethod
    def hetatm_parsing(pdb_line: str) -> str:
        return pdb_line[:17] + "LG1" + " " + "X   1" + pdb_line[26:]

    @property
    def repcom(self) -> str:
        retval = files_only_pdb(self.input_path)
        if self.af_model is not None:
            retval = [t for t in retval if t != self.af_model]
        if self.exclude_list is not None:
            retval = [t for t in retval if t not in self.exclude_list]
        retval = retval[0]
        print(f"Select {retval} to generate 'holo.pdb'")
        return retval

    @property
    def all_pdbs(self) -> list:
        retval = files_only_pdb(self.input_path)
        if self.exclude_list is not None:
            retval = [t for t in retval if t not in self.exclude_list]
        if self.native is not None:
            retval = [t for t in retval if t not in self.native.name]
        assert len(retval) > 0, "There are no PDBs"
        return retval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("--prefix")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path for save preprocessed files. If this arguemnt is not provided, the output path is the same as input path",
    )
    parser.add_argument("--out-af", default=None, help="Save path for pLDDT")
    parser.add_argument(
        "--ligand-name",
        default="LG1",
        type=str,
        help="Ligand name in complex PDB (residue name in PDB lines). Ligand names in each complex PDB files have to be same",
    )
    parser.add_argument(
        "--ligand-mol2",
        default=None,
        type=str,
        help="(Optinal) Ligand file with mol2 file format in input files directory. If this option were not given, ligand will be extracted from a complex PDB file",
    )
    parser.add_argument(
        "--complex-to-extract-ligand",
        default=None,
        type=str,
        help="(Optinal) Specifies the complex PDB to create the ligand mol2 file. If there is no ligand mol2 file, the ligand mol2 file is created from the complex PDB file. (Not necessary in most situations)",
    )
    parser.add_argument(
        "--add-h",
        action="store_true",
        help="(Optinal) Option given when all hydrogens are not explicitly present in the decoy's ligand.",
    )
    parser.add_argument(
        "-m",
        "--af-model",
        default=None,
        type=str,
        help="(Optinal) Option for model-docking (file name of AlphaFold model). If this option is not given, the input feature of pLDDT for the model will be set as ones",
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
        "--skip-ligand-processing",
        action="store_true",
        help="(Optinal) Option for skipping ligand processing (generate params)",
    )
    parser.add_argument(
        "--cross-docking",
        action="store_true",
        help="(Optinal) Option for calculating complex lDDT for the cross-docking task. It requires the native structure",
    )
    args = parser.parse_args()

    preprocess = Preprocess(
        input_path=Path(args.input),
        prefix=args.prefix,
        output_path=Path(args.output),
        af_plddt_save_path=Path(args.out_af),
        af_model=args.af_model,
        ligand_name=args.ligand_name,
        ligand_mol2=args.ligand_mol2,
        complex_to_extract_ligand=args.complex_to_extract_ligand,
        native=args.native,
        exclude=args.exclude,
        skip_ligand_processing=args.skip_ligand_processing,
        cross_docking=args.cross_docking,
    )
    preprocess.run()


if __name__ == "__main__":
    main()
