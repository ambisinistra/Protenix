# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
from pathlib import Path
from typing import Optional, List

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from protenix.data.data_pipeline import DataPipeline
from protenix.utils.file_io import dump_gzip_pickle

def gen_a_bioassembly_data(
    mmcif: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
) -> Optional[List[dict]]: # Изменил list на List
    """
    Generates bioassembly data from an mmCIF file and saves it to the specified output directory.

    Args:
        mmcif (Path): Path to the mmCIF file.
        bioassembly_output_dir (Path): Directory where the bioassembly data will be saved.
        cluster_file (Optional[Path]): Path to the cluster file, if available.
        distillation (bool, optional): Flag indicating whether to use the 'Distillation' setting. Defaults to False.

    Returns:
        Optional[List[dict]]: A list of sample indices if data is successfully generated, otherwise None.
    """
    if distillation:
        dataset = "Distillation"
    else:
        dataset = "WeightedPDB"

    # Эти строки требуют наличия класса DataPipeline и функции dump_gzip_pickle
    # В реальном коде убедитесь, что они импортированы или определены.
    try:
        sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
            mmcif, cluster_file, dataset
        )

        if sample_indices_list and bioassembly_dict:
            pdb_id = bioassembly_dict["pdb_id"]
            # save to output dir
            dump_gzip_pickle(bioassembly_dict, bioassembly_output_dir / f"{pdb_id}.pkl.gz")
            #logging.info(len(sample_indices_list))
            return sample_indices_list
        else:
             return None # Явно возвращаем None, если данные не были сгенерированы
    except Exception as e:
         # Добавим простую обработку ошибок, чтобы видеть, какие файлы вызывают проблемы при последовательной обработке
         print(f"Error processing {mmcif}: {e}")
         return None


# Переписанная функция gen_data_from_mmcifs без параллелизма
def gen_data_from_mmcifs( # Добавил суффикс _sequential к имени
    mmcif_list: List[Path], # Изменил list на List
    output_indices_csv: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    # Параметр num_workers удален, так как он не используется в последовательной версии
    # num_workers: int = 1,
):
    """
    Generates training data from a list of mmCIF files sequentially
    and saves the results to a CSV file.

    Args:
        mmcif_list (List[Path]): List of paths to mmCIF files.
        output_indices_csv (Path): Path to the output CSV file where the indices will be saved.
        bioassembly_output_dir (Path): Directory where the bioassembly output will be stored.
        cluster_file (Optional[Path]): Path to the cluster file. If None, clustering is not performed.
        distillation (bool, optional): Flag indicating whether to use the 'Distillation' setting. Defaults to False.
        # num_workers (int, optional): Removed as it's not used in sequential processing.
    """

    merged_results = [] # Список для сбора всех sample_indices

    # Заменяем Parallel processing на простой цикл for
    # Используем tqdm для отображения прогресса, как в оригинале
    for mmcif in tqdm(mmcif_list, total=len(mmcif_list)):
        sample_indices_list = gen_a_bioassembly_data(
            mmcif, bioassembly_output_dir, cluster_file, distillation
        )
        # Если gen_a_bioassembly_data вернула список (не None), добавляем его содержимое к merged_results
        if sample_indices_list:
            merged_results.extend(sample_indices_list) # Используем extend, чтобы добавить элементы списка, а не сам список

    # Остальная часть кода остается такой же
    df = pd.DataFrame(merged_results)

    df.to_csv(output_indices_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

def run_gen_data(
    input_path: Path,
    output_indices_csv: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    num_workers: int = 1,
):
    """
    Generates data from MMCIF files and saves the output to specified locations.

    Args:
        input_path (str): Path to the input directory containing MMCIF files or a text file listing MMCIF file paths.
        output_indices_csv (str): Path to the output CSV file where indices will be saved.
        bioassembly_output_dir (str): Directory where bioassembly outputs will be saved.
        cluster_file (Optional[str]): Path to the cluster file, if any.
        distillation (bool, optional): Flag indicating whether to use the 'Distillation' setting. Defaults to False.
        num_workers (int, optional): Number of worker processes to use. Defaults to 1.

    Raises:
        NotImplementedError: If the input path is not a directory or a text file.
    """

    input_path = Path(input_path)
    bioassembly_output_dir = Path(bioassembly_output_dir)
    output_indices_csv = Path(output_indices_csv)

    # create directory for output
    output_indices_csv.parent.mkdir(parents=True, exist_ok=True)
    bioassembly_output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        mmcif_list = list(input_path.glob("*.cif")) + list(input_path.glob("*.cif.gz"))
        mmcif_list = sorted(mmcif_list)
    elif input_path.suffix == ".txt":
        with open(input_path) as f:
            mmcif_list = [i.strip() for i in f.readlines()]
    else:
        raise NotImplementedError(f"Unsupported input path: {input_path}")

    gen_data_from_mmcifs(
        mmcif_list,
        output_indices_csv,
        bioassembly_output_dir,
        cluster_file,
        distillation,
        num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        default=None,
        help="Path to the input directory containing MMCIF files or a .txt file listing MMCIF file paths.",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=None,
        help="Path to the output CSV file where indices will be saved.",
    )
    parser.add_argument(
        "-b",
        "--bio_output_dir",
        type=Path,
        default=None,
        help="Directory where bioassembly outputs will be saved.",
    )
    parser.add_argument(
        "-c",
        "--cluster_file",
        type=Path,
        default=None,
        help="Path to the cluster txt file, if any",
    )

    parser.add_argument(
        "-d",
        "--distillation",
        action="store_true",
        help="Whether to use the 'Distillation' setting",
    )

    parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=1,
        help="Number of worker processes to use. Defaults to 1.",
    )

    args = parser.parse_args()

    run_gen_data(
        input_path=args.input_path,
        output_indices_csv=args.output_csv,
        bioassembly_output_dir=args.bio_output_dir,
        cluster_file=args.cluster_file,
        distillation=args.distillation,
        num_workers=args.n_cpu,
    )
