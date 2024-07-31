"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/07/31 15:54
  @Email: 2665109868@qq.com
  @function
"""

from os import makedirs, remove, walk
from os.path import join, isdir, exists, isfile
from tqdm import tqdm
import requests
from collections import defaultdict
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import gzip
import pandas as pd
import shutil
import time
import re
import ssl
import hashlib
import os
import tarfile
import urllib.request
import shutil
from tempfile import gettempdir
from timeit import default_timer as timer
from os.path import join, basename, isfile, isdir, dirname
from os import mkdir
import requests



def download_file(url, local_path):
    """ download a file to disk with ability to validate file checksum

    Parameters
    ----------
    url : str
        represents file full web url
    local_path : str
        represents full local path
    checksum : str
        represents the checksum of the file
    Returns
    -------
    str
        local path to downloaded file
    """

    # create a temporary file to download to
    tmp_dir = gettempdir()
    file_name = url.split('/')[-1]
    tmp_file = os.path.join(tmp_dir, file_name)

    urllib.request.urlretrieve(url, tmp_file)

    # move tmp file to desired file local path
    shutil.move(tmp_file, local_path)
    return local_path
def download_all_files(source_id,target_ids,id_name_map):
    """

    :param source_id:
    :param target_ids:
    """

    # 定义URL模板
    url_template = "https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id{source_id}/src{source_id}src{target_id}.txt.gz"
    filename_template = "drugbank_to_{db_name}.txt.gz"
    # 生成URL列表
    urls = [url_template.format(source_id=source_id, target_id=target_id) for target_id in target_ids]
    filenames = [filename_template.format(db_name = id_name_map[target_id]) for target_id in target_ids]
    urls.append('https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id1/src1src2.txt.gz')
    filenames.append('chembl_to_drugbank.txt.gz')

    for url,fileneme in zip(urls, filenames):
        try:
            local_path = download_file(url, '/project/reproduct_paper/biodblinker_supplement/drug_mapping_supplement_data/'+fileneme)
            print(local_path)
        except Exception as e:
            print(e)
    print('下载完成')

if __name__ == '__main__':

    source_id = 2
    target_ids = [31, 14, 6, 3, 17, 47, 37]
    id_name_map= {31:'bindingdb',14:'faers',6:'kegg compounds',3:'pdb',17:'pharmgkb',47:'rxnorm',37:'brenda'}
    download_all_files(source_id,target_ids,id_name_map)