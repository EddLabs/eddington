"""Module for saving content."""
import csv
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import openpyxl


def save_as_excel(
    content: List[Union[List[Any], np.ndarray]],
    output_directory: Union[str, Path],
    file_name: str,
    sheet: Optional[str] = None,
):
    """
    Save content to xlsx file.

    :param content: list of list, each represent a row in the excel file
    :type content: List[List[Any]] or numpy.ndarray
    :param output_directory: Path to the directory for the new excel file to be
     saved.
    :type output_directory: ``Path`` or ``str``
    :param file_name: The name of the file without suffix.
    :type file_name: str
    :param sheet: Optional. Name of the sheet that the data will be saved to.
    :type sheet: str or None
    """
    workbook = openpyxl.Workbook()
    worksheet = workbook.active

    if sheet:
        worksheet.title = sheet

    for row in content:
        worksheet.append(row)

    path = Path(output_directory) / f"{file_name}.xlsx"

    workbook.save(path)


def save_as_csv(
    content: List[Union[List[Any], np.ndarray]],
    file_name: str,
    output_directory: Union[str, Path],
):
    """
    Save content to csv file.

    :param content: list of list, each represent a row in the excel file
    :type content: List[List[Any]] or numpy.ndarray
    :param file_name: The name of the file without suffix.
    :type file_name: str
    :param output_directory:
     Path to the directory for the new excel file to be saved.
    :type output_directory: ``Path`` or ``str``
    """
    path = Path(output_directory / Path(f"{file_name}.csv"))

    with open(path, mode="w+", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(content)
