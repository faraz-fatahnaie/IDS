import pandas as pd
import numpy as np

from Cart2Pixel import Cart2Pixel
from ConvPixel import ConvPixel
from settings import *
from loader import parse_data


def main():

    param = {"Max_A_Size": 11, "Max_B_Size": 11, "Dynamic_Size": False, 'Method': 'tSNE',
             "seed": 1401,
             "dir": str(BASE_DIR.joinpath('Dataset2Image')), "Mode": "CNN2",  # Mode : CNN_Nature, CNN2
             "LoadFromPickle": False, "mutual_info": True,  # Mean or MI
             "hyper_opt_evals": 50, "epoch": 2, "No_0_MI": False,  # True -> Removing 0 MI Features
             "autoencoder": False, "cut": None, "enhanced_dataset": "gan"  # gan, smote, adasyn, ""None""
             }

    dataset_path = DATASET_DIR.joinpath('smote')
    dataset_catalog = {
        'train': dataset_path.joinpath('train'),
        'test': dataset_path.joinpath('test')}

    train_df = pd.read_csv(dataset_catalog['train'])
    x_train, y_train = parse_data(train_df, mode='df')
    #x_train = x_train.iloc[:200, :]
    #y_train = y_train.iloc[:200, :]
    print(f'train shape: x=>{x_train.shape}, y=>{y_train.shape}')
    y_train = y_train.to_numpy()

    test_df = pd.read_csv(dataset_catalog['test'])
    x_test, y_test = parse_data(test_df, mode='df')
    #x_test = x_test.iloc[:200, :]
    #y_test = y_test.iloc[:200, :]
    print(f'test shape: x=>{x_test.shape}, y=>{y_test.shape}')

    np.random.seed(param["seed"])
    print("transposing")
    # q["data"] is matrix T in paper (transpose of dataset without labels)
    # max_A_size, max_B_size is n and m in paper (the final size of generated image)
    # q["y"] is labels
    q = {"data": np.array(x_train.values).transpose(), "method": param["Method"],
         "max_A_size": param["Max_A_Size"], "max_B_size": param["Max_B_Size"], "y": y_train.argmax(axis=-1)}
    print(q["method"])
    print(q["max_A_size"])
    print(q["max_B_size"])

    # generate images
    XGlobal, image_model, toDelete = Cart2Pixel(q, q["max_A_size"], q["max_B_size"], param["Dynamic_Size"],
                                                mutual_info=param["mutual_info"], params=param, only_model=False)

    # saving images
    name = "_" + str(int(q["max_A_size"])) + "x" + str(int(q["max_B_size"]))
    if param["No_0_MI"]:
        name = name + "_No_0_MI"
    if param["mutual_info"]:
        name = name + "_MI"
    else:
        name = name + "_Mean"
    if image_model["custom_cut"] is not None:
        name = name + "_Cut" + str(image_model["custom_cut"])
    filename_train = "train" + name + ".npy"
    filename_test = "test" + name + ".npy"

    np.save(filename_train, XGlobal)
    print("Train Images generated and train images with labels are saved with the size of:", np.shape(XGlobal))

    # generate testing set image
    if param["mutual_info"]:
        x_test = x_test.drop(x_test.columns[toDelete], axis=1)

    x_test = np.array(x_test).transpose()
    print("generating Test Images for X_test with size ", x_test.shape)

    if image_model["custom_cut"] is not None:
        XTestGlobal = [ConvPixel(x_test[:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                       image_model["A"], image_model["B"], custom_cut=range(0, image_model["custom_cut"]))
                       for i in range(0, x_test.shape[1])]
    else:
        XTestGlobal = [ConvPixel(x_test[:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                       image_model["A"], image_model["B"])
                       for i in range(0, x_test.shape[1])]

    np.save(filename_test, XTestGlobal)
    print("Test Images generated and test images with labels are saved with the size of:", np.shape(XTestGlobal))

    return 1


if __name__ == '__main__':
    main()
