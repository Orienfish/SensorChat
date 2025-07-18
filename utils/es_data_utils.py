import numpy as np


# The ID order of the users in ExtraSensory
all_user_ids = ['5EF64122-B513-46AE-BCF1-E62AAC285D2C', 'B9724848-C7E2-45F4-9B3F-A1F38D864495', '27E04243-B138-4F40-A164-F40B60165CF3', '74B86067-5D4B-43CF-82CF-341B76BEA0F4', '7D9BB102-A612-4E2A-8E22-3159752F55D8', 'BEF6C611-50DA-4971-A040-87FB979F3FC1', '40E170A7-607B-4578-AF04-F021C3B0384A', 'C48CE857-A0DD-4DDB-BEA5-3A25449B2153', 'A5A30F76-581E-4757-97A2-957553A2C6AA', 'CDA3BBF7-6631-45E8-85BA-EEB416B32A3C', '61976C24-1C50-4355-9C49-AAE44A7D09F6', 'FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF', '806289BC-AD52-4CC1-806C-0CDB14D65EB6', '136562B6-95B2-483D-88DC-065F28409FD2', '8023FE1A-D3B0-4E2C-A57A-9321B7FC755F', '11B5EC4D-4133-4289-B475-4E737182A406', '5152A2DF-FAF3-4BA8-9CA9-E66B32671A53', 'ECECC2AB-D32F-4F90-B74C-E12A1C69BBE2', 'BE3CA5A6-A561-4BBD-B7C9-5DF6805400FC', 'A5CDF89D-02A2-4EC1-89F8-F534FDABDD96', '61359772-D8D8-480D-B623-7C636EAD0C81', '2C32C23E-E30C-498A-8DD2-0EFB9150A02E', '86A4F379-B305-473D-9D83-FC7D800180EF', '33A85C34-CFE4-4732-9E73-0A7AC861B27A', 'B7F9D634-263E-4A97-87F9-6FFB4DDCB36C', '83CF687B-7CEC-434B-9FE8-00C3D5799BE6', 'A76A5AF5-5A93-4CF2-A16E-62353BB70E8A', '99B204C0-DD5C-4BB7-83E8-A37281B8D769', '59EEFAE0-DEB0-4FFF-9250-54D2A03D0CF2', '24E40C4C-A349-4F9F-93AB-01D00FB994AF', '59818CD2-24D7-4D32-B133-24C2FE3801E5', 'E65577C1-8D5D-4F70-AF23-B3ADB9D3DBA3', 'A7599A50-24AE-46A6-8EA6-2576F1011D81', '3600D531-0C55-44A7-AE95-A7A38519464E', '098A72A5-E3E5-4F54-A152-BBDA0DF7B694', '5119D0F8-FCA8-4184-A4EB-19421A40DE0D', '81536B0A-8DBF-4D8A-AC24-9543E2E4C8E0', '00EABED2-271D-49D8-B599-1D4A09240601', '797D145F-3858-4A7F-A7C2-A4EB721E133C', '1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842', '1538C99F-BA1E-4EFB-A949-6C7C47701B20', 'CA820D43-E5E2-42EF-9798-BE56F776370B', '481F4DD2-7689-43B9-A2AA-C8772227162B', 'CCAF77F0-FABB-4F2F-9E24-D56AD0C5A82F', '0BFC35E2-4817-4865-BFA7-764742302A2D', '1155FF54-63D3-4AB2-9863-8385D0BD0A13', '9DC38D04-E82E-4F29-AB52-B476535226F2', 'B09E373F-8A54-44C8-895B-0039390B859F', '7CE37510-56D0-4120-A1CF-0E23351428D2', 'D7D20E2E-FC78-405D-B346-DBD3FD8FC92B', 'F50235E0-DD67-4F2A-B00B-1F31ADA998B9', '4FC32141-E888-4BFF-8804-12559A491D8C', '665514DE-49DC-421F-8DCB-145D0B2609AD', '4E98F91F-4654-42EF-B908-A3389443F2E7', '9759096F-1119-4E19-A0AD-6F16989C7E1C', '96A358A0-FFF2-4239-B93E-C7425B901B47', '0A986513-7828-4D53-AA1F-E02D6DF9561B', 'CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC', '78A91A4E-4A51-4065-BDA7-94755F0BB3BB', '0E6184E1-90C0-48EE-B25A-F1ECB7B9714E']


merged_label_names = [
        'SITTING', 
        'LYING_DOWN',
        'OR_standing',
        'FIX_walking',
        'BICYCLING',
        'PHONE_ON_TABLE',
        'PHONE_IN_POCKET',
        'PHONE_IN_HAND',
        'PHONE_IN_BAG',
        'WITH_CO-WORKERS',
        'WITH_FRIENDS',
        'OR_outside',
        'OR_indoors',
        'AT_SCHOOL',
        'LOC_home',
        'LOC_main_workplace',
        'IN_CLASS',
        'IN_A_MEETING',
        'TOILET',
        'COOKING',
        'SLEEPING',
        'EATING',
        'BATHING_-_SHOWER',
        'LAB_WORK',
        'COMPUTER_WORK',
        'SURFING_THE_INTERNET',
        'OR_exercise',
        'TALKING',
        'WATCHING_TV',
        'On a vehicle',
        'CLEANING',
        'GROOMING',
    ]

def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'walking';
    if label == 'FIX_running':
        return 'running';
    if label == 'LOC_main_workplace':
        return 'at main workplace';
    if label == 'OR_indoors':
        return 'indoors';
    if label == 'OR_outside':
        return 'outside';
    if label == 'LOC_home':
        return 'at home';
    if label == 'FIX_restaurant':
        return 'at a restaurant';
    if label == 'OR_exercise':
        return 'exercising';
    if label == 'LOC_beach':
        return 'at the beach';
    if label == 'OR_standing':
        return 'standing';
    if label == 'WATCHING_TV':
        return 'watching TV'
    if label == 'DRINKING__ALCOHOL_':
        return 'drinking alcohol'
    if label == 'BATHING_-_SHOWER':
        return 'bathing or showering'
    if label == 'TOILET':
        return 'in toilet'
    if label == 'LAB_WORK':
        return 'doing lab work'
    if label == 'COMPUTER_WORK':
        return 'doing computer work'
    
    if label.endswith('_'):
        label = label[:-1] + ')';
        pass;
    
    label = label.replace('__',' (').replace('_',' ');
    label = label.lower();
    label = label.replace('i m','I\'m');

    # get rid of phone
    #if 'phone' in label:
    #    label = label.replace('phone ', '')

    return label;

import gzip;
from io import StringIO, BytesIO

def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index(b'\n')]
    columns = headline.split(b',')

    # The first column should be timestamp:
    assert columns[0] == b'timestamp'
    # The last column should be label_source:
    assert columns[-1] == b'label_source'

    # Search for the column of the first label:
    for (ci, col) in enumerate(columns):
        if col.startswith(b'label:'):
            first_label_ind = ci
            break
        pass

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind]
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1]
    for (li, label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith(b'label:')
        label_names[li] = label.replace(b'label:', b'')
        pass

    return (feature_names, label_names)


def parse_body_of_csv(csv_str, n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(BytesIO(csv_str), delimiter=',', skiprows=1)

    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:, 0].astype(int)

    # Read the sensor features:
    X = full_table[:, 1:(n_features + 1)]

    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:, (n_features + 1):-1]  # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat)  # M is the missing label matrix
    Y = np.where(M, 0, trinary_labels_mat) > 0.  # Y is the label matrix

    return (X, Y, M, timestamps)

'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''
def read_user_data(user_data_file):
    # user_data_file = os.path.join(DATA_PATH, f"{uuid}.features_labels.csv.gz")
    # Read the entire csv file of the user:
    with gzip.open(user_data_file, 'rb') as fid:
        csv_str = fid.read()
        pass
    
    (feature_names, label_names) = parse_header_of_csv(csv_str)
    n_features = len(feature_names)
    (X, Y, M, timestamps) = parse_body_of_csv(csv_str, n_features)

    return (X, Y, M, timestamps, feature_names, label_names)