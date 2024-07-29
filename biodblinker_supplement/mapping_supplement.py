"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/07/29 15:54
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
def download_file_md5_check(download_url, filepath, username=None, password=None):
    """ Download file if not existing and have valid md5

    Parameters
    ----------
    download_url : str
        file download url
    filepath : str
        file full local path
    """
    filename = basename(filephath)
    md5_dp = join(dirname(filepath), "checksum")
    md5_filepath = join(md5_dp, filename + ".md5")

    require_download = False
    if isfile(filepath) and isfile(md5_filepath):
        file_computed_md5 = get_file_md5(filepath)
        file_saved_md5_fd = open(md5_filepath, "r")
        file_saved_md5 = file_saved_md5_fd.read().strip()
        file_saved_md5_fd.close()
        print(inf_sym + "file (%-40s) exists." % filename, end="", flush=True)
        if file_computed_md5 == file_saved_md5:
            print(hsh_sym + " MD5 hash (%s) is correct%s. No download is required." % (file_saved_md5, done_sym))
        else:
            require_download = True
            print(hsh_sym + " MD5 hash (%s) is inconsistent%s. Re-downloading ..." % (file_computed_md5, fail_sym))
    else:
        require_download = True

    if require_download:
        print(dwn_sym + "downloading file (%-40s) ..." % filename, end="", flush=True)
        start = timer()
        if username is None or password is None:
            download_file(download_url, filepath)
            # pass
        else:
            # download_file_with_auth(download_url, filepath, username, password)
            pass
        download_time = timer() - start
        print(done_sym + " %1.2f Seconds." % download_time, end="", flush=True)
        file_computed_md5 = get_file_md5(filepath)
        mkdir(md5_dp) if not isdir(md5_dp) else None
        md5_fd = open(md5_filepath, "w")
        md5_fd.write(file_computed_md5)
        md5_fd.close()
        print(hsh_sym + "md5 hash saved (%s)." % file_computed_md5, flush=True)
def _download_cellosaurus_files(self, source_dir):
    """ Download cellosaurus file

    Parameters
    ----------
    sources_dir : str
        the path to save the files

    Returns
    -------
    str
        the path to the cellosaurus file
    """
    cello_file_url = 'ftp://ftp.expasy.org/databases/cellosaurus/cellosaurus.xml'
    cello_filepath = join(source_dir, "cellosaurus.xml")
    download_file_md5_check(cello_file_url, cello_filepath)
    return cello_filepath

def download_file(url, local_path, checksum=None):
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

    # check the checksum if provided
    if not (checksum is None):
        # compute the checksum of the file
        downloaded_file_checksum = get_file_md5(tmp_file)
        if downloaded_file_checksum != checksum:
            raise ValueError("invalid file checksum [%s] for file: %s" % (downloaded_file_checksum, url))

    # move tmp file to desired file local path
    shutil.move(tmp_file, local_path)
    return local_path