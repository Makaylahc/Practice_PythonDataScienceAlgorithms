'''
Here is the retrieve_emails() function that is being supplied for Sections A/C, only. 
This function has NOT been supplied to students in Section B, and should not be shared.

AUTHOR: Oliver W. Layton, Spring 2020. Thank you, Oliver!
'''
import re
import os
import numpy as np

def retrieve_emails(inds, email_path='data/enron/'):
    '''Obtain the text of emails at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
        (counting does NOT reset when switching to emails of another class).
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    '''
    emails = []

    class_names = os.listdir(email_path)
    class_names = [filename for filename in class_names if os.path.isdir(os.path.join(email_path, filename))]
    print(f'Discovered class names: {class_names}')

    i = 0

    for c in range(len(class_names)):
        class_dir = os.path.join(email_path, class_names[c])
        email_filenames = os.listdir(class_dir)
        print(f'Processing {class_dir}...')

        for email_file in email_filenames:
            curr_file = os.path.join(class_dir, email_file)
            if np.any(i == inds):
                with open(curr_file, 'r', encoding='latin-1') as fp:
                    try:
                        mail_text = fp.readlines()
                        mail_text = ''.join(mail_text)
                    except UnicodeDecodeError:
                        print(curr_file)
                emails.append(mail_text)
            i += 1
    return emails