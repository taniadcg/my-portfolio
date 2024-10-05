# Folders Synchronization
Synchronizes two folders 

## Table of Contents
1. [Description](#description)
2. [Program and Version](#program-and-version)
3. [Features](#features)
4. [Usage](#usage)
5. [Author](#author)
6. [Contact](#contact)
7. [FAQ](#faq)

## Description
The Folder Synchronizer program ensures that the replica folder is always an exact copy of the source folder.

## Program and Version
- **Programming Language**: Python
- **Version**: 3.9.7
- **Development Environment**: Spyder (Anaconda 3)

## Features
- **Full Synchronization**: Makes the replica folder an identical copy of the source folder.
- **Add Files**: Copies any new files from the source to the replica.
- **Delete Files**: Removes files from the replica if they no longer exist in the source.
- **Update Files**: Updates files in the replica if they have been changed in the source.
- **Handle Subfolders**: Ensures all subfolders and their contents are synchronized.

## Usage
How to run the project:  
  python sync_folders_task.py /path/to/source /path/to/replica syncInterval /path/to/logfile.log
  
  **Note:** Make sure to specify the 'syncInterval' in seconds.

## Author:
Tânia Gonçalves

## Contact:
For further information or support, contact gtaniadc@gmail.com.

## FAQ
**Q1: What is the purpose of this program?**  
**A1:** The program synchronizes a source folder with a replica folder, ensuring the replica is an exact copy of the source.

**Q2: How often does the synchronization occur?**  
**A2:** The synchronization occurs at intervals specified by the 'syncInterval' parameter, which must be entered in seconds.

**Q3: What happens if a file is deleted from the source folder?**  
**A3:** If a file is deleted from the source folder, it will also be deleted from the replica folder during the next synchronization.

**Q4: Are subfolders also synchronized?**  
**A4:** Yes, all subfolders and their contents are synchronized along with the main folder.

**Q5: What if a file in the replica folder is modified?**  
**A5:** If a file in the replica folder is modified, it will be overwritten by the corresponding file from the source folder during the next synchronization.

**Q6: What programming language and environment are used for this program?**  
**A6:** The program is written in Python 3.9.7 and developed using Spyder in the Anaconda 3 environment.

**Q7: Can I configure the synchronization process?**  
**A7:** Yes, you can configure the synchronization process by setting the appropriate parameters in the program, such as the source folder, replica folder, and syncInterval.

**Q8: How do I run the program?**  
**A8:** You can run the program by executing the following command:  
python sync_folders_task.py /path/to/source /path/to/replica syncInterval /path/to/logfile.log  
Make sure to specify the 'syncInterval' in seconds.

**Q9: Who should I contact for further information or support?**  
**A9:** For further information or support, you can contact the project maintainer at gtaniadc@gmail.com.
