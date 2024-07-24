import os
import numpy as np
from typing import Optional
from pathlib import Path

from denoiser.utils.train_utils import files_only_pdb


class ExtractAFpLDDT:
    def __init__(
        self,
        input_path: Path,
        prefix: str,
        save_path: Optional[Path] = None,
        af_model: Optional[str] = None,
    ):
        self.input_path = input_path
        self.prefix = prefix
        if save_path is None:
            save_path = input_path
        self.save_path = save_path
        self.af_model = af_model

    def save_plddt(self) -> None:
        res_conf = []
        with open(self.target) as f:
            for line in f:
                if line[:4] == "ATOM" and line[13:15] == "CA":
                    if self.af_model:
                        conf = float(line.split()[-2])
                    else:
                        conf = 1.0
                    # Scaling pLDDT to 0~1
                    if conf > 1:
                        conf = round(conf / 100, 2)
                    res_conf.append(conf)
        res_conf = np.reshape(np.array(res_conf), (-1, 1))
        np.save(self.save_path.joinpath(self.prefix + "_conf.npy"), res_conf)

    @property
    def target(self) -> str:
        if self.af_model:
            retval = self.input_path.joinpath(self.af_model)
        else:
            repstr = files_only_pdb(self.input_path)[0]
            retval = self.input_path.joinpath(repstr)
        print(
            f"Select {retval} to extract pLDDT and save. If no AF model is provided, plDDT is set to 1."
        )
        return retval
