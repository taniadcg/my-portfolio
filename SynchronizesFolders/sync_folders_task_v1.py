# -*- coding: utf-8 -*-
"""
Script Name: sync_folders_task_v1.py
Description: This script synchronizes the contents of a source folder with a replica folder.
Author: Tânia Gonçalves
Date: 20/09/2024
Version: 1.0
Usage: python sync_folders_task_v1.py /path/to/source /path/to/replica syncInterval /path/to/logfile.log

"""

# Import Libraries: os: file and directory operations, shutil: copy files and directories, argparse: parses command-line arguments,
# time: adds delays, logging: records log messages, hashlib:calculating file checksums for comparison
import os
import shutil
import argparse
import time
import logging
import hashlib

# Function parse_arguments: Command-line arguments for source folder, replica folder, interval, and log file are parsed
def parse_arguments():
    parser = argparse.ArgumentParser(description='Synchronize two folders at a specified interval.')
    parser.add_argument('source_folder', type=str, help='Path to the source folder')
    parser.add_argument('replica_folder', type=str, help='Path to the replica folder')
    parser.add_argument('interval', type=int, help='Synchronization interval in seconds')
    parser.add_argument('log_file', type=str, help='Path to the log file')
    return parser.parse_args()

# Function calculate_md5: computes and returns the MD5 checksum of a given file, enabling file content verification and comparison
def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Function sync_folders: Copies files and directories from the source to the replica folder           
def sync_folders(source, replica, f_logger):
    
    # Create any missing directories in the replica folder
    if not os.path.exists(replica):
        os.makedirs(replica)
        
    # Track existing items in replica - Keeps track of items that have been processed to identify which items should be deleted from the destination
    existing_items = set(os.listdir(replica))
        
    # Lists all files and directories in the source folder
    for item in os.listdir(source):
        
        #Constructs the full paths to the current item in the source and replica folders
        source_path = os.path.join(source, item)
        replica_path = os.path.join(replica, item)
        
        # Checks if the current item is a directory.
        if os.path.isdir(source_path):
            print('Recursively synchronize subdirectories')
            sync_folders(source_path, replica_path, f_logger)  # Recursively synchronize subdirectories
            if not os.path.exists(replica_path):
                shutil.copytree(source_path, replica_path) # Copies the directory 
                f_logger.info(f'Copied {source_path} to {replica_path}')
                print(f'IF - Copied {source_path} to {replica_path}')
        else:
            # Compare MD5 hashes to check if the file needs to be copied
            if not os.path.exists(replica_path) or calculate_md5(source_path) != calculate_md5(replica_path):
                shutil.copy2(source_path, replica_path) # Copies the file
                f_logger.info(f'Copied {source_path} to {replica_path}')
                print(f'ELSE - Copied {source_path} to {replica_path}')
            else:
                print(f'Skipped {source_path}, no changes detected')
     
        # Remove item from existing items set
        existing_items.discard(item)
        
    # Delete items in replica that are not in source
    for item in existing_items:
        item_path = os.path.join(replica, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            f_logger.info(f'Deleted directory {item_path}')
            print(f'IF - Deleted directory {item_path}')
        else:
            os.remove(item_path)
            f_logger.info(f'Deleted file {item_path}')
            print(f'ELSE - Deleted file {item_path}')
    
    f_logger.info(f'Synchronized {source} to {replica}')
    print(f'Synchronized {source} to {replica}')

# Function Main: Defines the source and replica folders and calls the sync_folders function
def main():

    args = parse_arguments()
    
    # Create a logger
    logger = logging.getLogger(__name__)

    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.INFO)

    # Create file handler
    FileOutputHandler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    FileOutputHandler.setFormatter(formatter)
    logger.addHandler(FileOutputHandler)
    
    # Infinite Loop: Continuously synchronizes the source folder to the replica folder at the specified interval, logging each synchronization
    while True:
        sync_folders(args.source_folder, args.replica_folder, logger)
        time.sleep(args.interval)

if __name__ == '__main__':
    main()
