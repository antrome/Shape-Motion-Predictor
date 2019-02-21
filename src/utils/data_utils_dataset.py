import os
from os.path import join, exists, abspath, dirname
import numpy as np

Action_Name = {
    '2': 'Directions',
    '3': 'Discussion',
    '4': 'Eating',
    '5': 'Greeting',
    '6': 'Phoning',
    '7': 'Posing',
    '8': 'Purchases',
    '9': 'Sitting',
    '10': 'SittingDown',
    '11': 'Smoking',
    '12': 'Photo',
    '13': 'Waiting',
    '14': 'Walking',
    '15': 'WalkDog',
    '16': 'WalkTogether',
}

SUBACT_dict = {
    ('Directions', 1): [" 1", ""],
    ('Directions', 5): [" 1", " 2"],
    ('Directions', 6): [" 1", ""],
    ('Directions', 7): [" 1", ""],
    ('Directions', 8): [" 1", ""],
    ('Directions', 9): [" 1", ""],
    ('Directions', 11): [" 1", ""],
    #
    ('Discussion', 1): [" 1", ""],
    ('Discussion', 5): [" 2", " 3"],
    ('Discussion', 6): [" 1", ""],
    ('Discussion', 7): [" 1", ""],
    ('Discussion', 8): [" 1", ""],
    ('Discussion', 9): [" 1", " 2"],
    ('Discussion', 11): [" 1", " 2"],
    #
    ('Eating', 1): [" 2", ""],
    ('Eating', 5): [" 1", ""],
    ('Eating', 6): [" 1", " 2"],
    ('Eating', 7): [" 1", ""],
    ('Eating', 8): [" 1", ""],
    ('Eating', 9): [" 1", ""],
    ('Eating', 11): [" 1", ""],
    #
    ('Greeting', 1): [" 1", ""],
    ('Greeting', 5): [" 1", " 2"],
    ('Greeting', 6): [" 1", ""],
    ('Greeting', 7): [" 1", ""],
    ('Greeting', 8): [" 1", ""],
    ('Greeting', 9): [" 1", ""],
    ('Greeting', 11): [" 2", ""],
    #
    ('Phoning', 1): [" 1", ""],
    ('Phoning', 5): [" 1", ""],
    ('Phoning', 6): [" 1", ""],
    ('Phoning', 7): [" 2", ""],
    ('Phoning', 8): [" 1", ""],
    ('Phoning', 9): [" 1", ""],
    ('Phoning', 11): [" 2", " 3"],
    #
    ('Photo', 1): [" 1", ""],
    ('Photo', 5): ["", " 2"],
    ('Photo', 6): [" 1", ""],
    ('Photo', 7): [" 1", ""],
    ('Photo', 8): [" 1", ""],
    ('Photo', 9): [" 1", ""],
    ('Photo', 11): [" 1", ""],
    #
    ('Posing', 1): [" 1", ""],
    ('Posing', 5): [" 1", ""],
    ('Posing', 6): [" 2", ""],
    ('Posing', 7): [" 1", ""],
    ('Posing', 8): [" 1", ""],
    ('Posing', 9): [" 1", ""],
    ('Posing', 11): [" 1", ""],
    #
    ('Purchases', 1): [" 1", ""],
    ('Purchases', 5): [" 1", ""],
    ('Purchases', 6): [" 1", ""],
    ('Purchases', 7): [" 1", ""],
    ('Purchases', 8): [" 1", ""],
    ('Purchases', 9): [" 1", ""],
    ('Purchases', 11): [" 1", ""],
    #
    ('Sitting', 1): [" 1", " 2"],
    ('Sitting', 5): [" 1", ""],
    ('Sitting', 6): [" 1", " 2"],
    ('Sitting', 7): [" 1", ""],
    ('Sitting', 8): [" 1", ""],
    ('Sitting', 9): [" 1", ""],
    ('Sitting', 11): [" 1", ""],
    #
    ('SittingDown', 1): [" 2", ""],
    ('SittingDown', 5): ["", " 1"],
    ('SittingDown', 6): [" 1", ""],
    ('SittingDown', 7): [" 1", ""],
    ('SittingDown', 8): [" 1", ""],
    ('SittingDown', 9): [" 1", ""],
    ('SittingDown', 11): [" 1", ""],
    #
    ('Smoking', 1): [" 1", ""],
    ('Smoking', 5): [" 1", ""],
    ('Smoking', 6): [" 1", ""],
    ('Smoking', 7): [" 1", ""],
    ('Smoking', 8): [" 1", ""],
    ('Smoking', 9): [" 1", ""],
    ('Smoking', 11): [" 2", ""],
    #
    ('Waiting', 1): [" 1", ""],
    ('Waiting', 5): [" 1", " 2"],
    ('Waiting', 6): [" 3", ""],
    ('Waiting', 7): [" 1", " 2"],
    ('Waiting', 8): [" 1", ""],
    ('Waiting', 9): [" 1", ""],
    ('Waiting', 11): [" 1", ""],
    #
    ('WalkDog', 1): [" 1", ""],
    ('WalkDog', 5): [" 1", ""],
    ('WalkDog', 6): [" 1", ""],
    ('WalkDog', 7): [" 1", ""],
    ('WalkDog', 8): [" 1", ""],
    ('WalkDog', 9): [" 1", ""],
    ('WalkDog', 11): [" 1", ""],
    #
    ('Walking', 1): [" 1", ""],
    ('Walking', 5): [" 1", ""],
    ('Walking', 6): [" 1", ""],
    ('Walking', 7): [" 1", " 2"],
    ('Walking', 8): [" 1", ""],
    ('Walking', 9): [" 1", ""],
    ('Walking', 11): [" 1", ""],
    #
    ('WalkTogether', 1): [" 1", ""],
    ('WalkTogether', 5): [" 1", ""],
    ('WalkTogether', 6): [" 1", ""],
    ('WalkTogether', 7): [" 1", ""],
    ('WalkTogether', 8): [" 1", " 2"],
    ('WalkTogether', 9): [" 1", ""],
    ('WalkTogether', 11): [" 1", ""],
}

Subject_Gender = {
    '1': 'female',
    '5': 'female',
    '6': 'male',
    '7': 'female',
    '8': 'male',
    '9': 'male',
    '11': 'male'
}




def define_paths(SUBJECT_ID,ACTION_ID,SUBACTION_ID,CAMERA_ID,IMG_DIR,MODEL_DIR,DATA_SHAPE_DIR,DATA_2D_DETECT_DIR):

    # Model file:
    gender = Subject_Gender[SUBJECT_ID]

    if gender == 'neutral':
        MODEL_FILE = join(MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    if gender == 'female':
        MODEL_FILE = join(MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    if gender == 'male':
        MODEL_FILE = join(MODEL_DIR, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    # Directories for the Sequence of images for all cameras (1-4)
    IMG_SEQUENCE_DIR =  IMG_DIR + 'S{0}'.format(int(SUBJECT_ID))+ \
                    '/VideoFrames/Act' + '{0}'.format(int(ACTION_ID)).zfill(2) + \
                    '_Subact' + '{0}'.format(int(SUBACTION_ID)).zfill(2) + \
                    '_Cam' + '{0}'.format(int(CAMERA_ID)).zfill(2) + '/'

    # Data Shape file
    Act_id=Action_Name[ACTION_ID]
    SUBACT_TXT = SUBACT_dict[Act_id, int(SUBJECT_ID)]
    if SUBACTION_ID=='1':
        DATA_SHAPE_FILE = DATA_SHAPE_DIR + 'S{0}'.format(int(SUBJECT_ID)) + '/' + Act_id + SUBACT_TXT[0] + '.pkl'
    elif SUBACTION_ID=='2':
        DATA_SHAPE_FILE = DATA_SHAPE_DIR + 'S{0}'.format(int(SUBJECT_ID)) + '/' + Act_id + SUBACT_TXT[1] + '.pkl'


    # Data Pose file
    DATA_2D_DETECT_DIR_ = DATA_2D_DETECT_DIR  + 'S' + '{0}'.format(int(SUBJECT_ID)).zfill(2) + \
                           '/Act' + '{0}'.format(int(ACTION_ID)).zfill(2) + \
                           '/Subact' + '{0}'.format(int(SUBACTION_ID)).zfill(2) + \
                           '/Cam' + '{0}'.format(int(CAMERA_ID)).zfill(2) + '/'


    #Results
    OUTPUT_DIR_DATA = 'H36M_Parsed' + '/data/' + 'S' + '{0}'.format(int(SUBJECT_ID)).zfill(2) + \
                           '/Act' + '{0}'.format(int(ACTION_ID)).zfill(2) + \
                           '/Subact' + '{0}'.format(int(SUBACTION_ID)).zfill(2) + \
                           '/Cam' + '{0}'.format(int(CAMERA_ID)).zfill(2) + '/'

    OUTPUT_DIR_IMG = 'H36M_Parsed' + '/images/' + 'S' + '{0}'.format(int(SUBJECT_ID)).zfill(2) + \
                           '/Act' + '{0}'.format(int(ACTION_ID)).zfill(2) + \
                           '/Subact' + '{0}'.format(int(SUBACTION_ID)).zfill(2) + \
                           '/Cam' + '{0}'.format(int(CAMERA_ID)).zfill(2) + '/'




    return MODEL_FILE, IMG_SEQUENCE_DIR, DATA_SHAPE_FILE, DATA_2D_DETECT_DIR_, OUTPUT_DIR_DATA, OUTPUT_DIR_IMG



def rotate_and_translate(R,T,X):
    Xc = np.matmul(R, X.transpose()) + T
    return Xc.transpose()

def project_points(A, R, T, X):
    Xc = rotate_and_translate(R,T,X)

    Uph = np.matmul(A, Xc.transpose())
    Uph = Uph.transpose()

    njoints = X.shape[0]
    Up = np.zeros([njoints,2])
    Up[:,0] = Uph[:,0]/Uph[:,2]
    Up[:,1] = Uph[:,1]/Uph[:,2]

    return Up, Xc












