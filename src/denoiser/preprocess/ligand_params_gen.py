import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import re

from denoiser.utils.chem_utils import mol2_atom_process, PDBLineParser as PDBLP


PARAM = "/home/hpark/programs/generic_potential/mol2genparams.py"


def is_float(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


class LigandParams:
    def __init__(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        ligand_mol2: Optional[str] = None,
        save_original_mol2: Optional[Path] = None,
        ligand_name: str = "LG1",
        add_h: bool = False,
    ):
        self.input_path = input_path.absolute()
        if output_path is None:
            output_path = input_path
        self.output_path = output_path.absolute()
        self.ligand_mol2 = ligand_mol2
        if input_path == output_path:
            assert save_original_mol2 is not None
        if save_original_mol2 is not None:
            save_original_mol2 = save_original_mol2.absolute()
        self.save_original_mol2 = save_original_mol2
        self.ligand_name = ligand_name
        self.add_h = add_h

    def save_params(self, complex_pdb: Optional[Path] = None) -> None:
        cur_dir = os.getcwd()
        # os.chdir(self.input_path)
        os.chdir(self.output_path)
        if complex_pdb is None:
            complex_pdb = self.output_path.joinpath("holo.pdb")
        print("Extract Ligand from", complex_pdb)

        if self.ligand_mol2 is None:
            self.extract_ligand(complex_pdb, self.ligand_name)
            self.pdb2mol2(self.add_h)
        else:
            if self.input_path == self.output_path:
                shutil.copy(
                    self.ligand_mol2, self.save_original_mol2.joinpath(self.ligand_mol2)
                )
            save_path = self.output_path.joinpath(self.ligand_mol2)
            mol2_atom_process(self.ligand_mol2, save_path)

        self.add_charge()
        self.generate_params()

        self.delete_unused_rename()

        os.chdir(cur_dir)

    def extract_ligand(self, complex_pdb: Path, ligand_name: str) -> None:
        ligand = self.output_path.joinpath("LG.pdb")
        with open(ligand, "w") as f1:
            with open(complex_pdb) as f2:
                for line in f2:
                    if line.startswith("HETATM"):
                        if PDBLP.residue_name(line) == ligand_name:
                            if is_float(line.split()[-1]):
                                atom = PDBLP.atom_name(line)
                                atom = re.sub(r"[0-9]", "", atom)
                                if atom.startswith("C") and atom.lower() != "cl":
                                    atom = "C"
                                elif len(atom) != 1 and atom.startswith("O"):
                                    atom = "O"
                                elif len(atom) != 1 and atom.startswith("H"):
                                    atom = "H"
                                elif len(atom) != 1 and atom.startswith("N"):
                                    atom = "N"

                                pad = (78 - len(line) - len(atom) + 1) * " "
                                line = line.strip() + pad + atom + "\n"
                            f1.write(line)
                    if line.startswith("CONECT"):
                        f1.write(line)

    def pdb2mol2(self, add_h: bool = False) -> None:
        input_path = str(self.output_path / "LG.pdb")
        save_mol2 = str(self.output_path / "LG.mol2")
        args = ["obabel", input_path, "-O", save_mol2]
        if add_h:
            args += ["-p", "7"]
        subprocess.run(args)

    def add_charge(self) -> None:
        def get_mol2_anames(mol2_file):
            check = 0
            a_types = []
            with open(mol2_file) as f:
                for line in f:
                    if line.startswith("@<TRIPOS>ATOM"):
                        check = 1
                        continue
                    if check and line.startswith("@<TRIPOS>"):
                        break
                    if check:
                        a_type = line.split()[1]
                        a_types.append(a_type)
            return a_types

        input_mol2 = (
            self.output_path / "LG.mol2"
            if self.ligand_mol2 is None
            else self.output_path / self.ligand_mol2
        )
        output_mol2 = self.output_path / "LG.mol2"
        orig_names = get_mol2_anames(input_mol2)
        args = ["obabel", input_mol2, "-O", output_mol2, "--partialcharge", "mmff94"]
        subprocess.run(args)

        # Replace atom names
        with open(output_mol2) as f:
            outlines = f.readlines()

        check, idx = 0, 0
        for i, line in enumerate(outlines):
            if line.startswith("@<TRIPOS>ATOM"):
                check = 1
                continue
            if check and line.startswith("@<TRIPOS>"):
                break
            if check:
                orig_n = orig_names[idx]
                outlines[i] = line[:8] + orig_n + line[8 + len(orig_n) :]
                idx += 1

        with open(output_mol2, "w") as f:
            for line in outlines:
                f.write(line)

    def generate_params(self) -> None:
        # conda_env = "/opt/conda/envs/gp/bin/python3.9"
        conda_env = "python3"
        input_mol2 = self.output_path / "LG.mol2"
        args = [
            conda_env,
            PARAM,
            "-s",
            str(input_mol2),
            "--outdir",
            str(self.output_path),
            "2>/dev/null",
        ]
        subprocess.run(args)

    def delete_unused_rename(self) -> None:
        if "LG.params" not in os.listdir(self.output_path):
            sys.exit("LG.params is not generated. Check the PDB files")
        remove_files = ["LG.mol2", "LG_0001.pdb"]
        if self.ligand_mol2 is None:
            remove_files += ["LG.pdb"]
        for f in remove_files:
            t = self.output_path / f
            if os.path.exists(t):
                os.remove(t)
