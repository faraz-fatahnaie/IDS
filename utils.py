
def parse_data(df, mode='np'):
    classes = ['Dos', 'Probe', 'R2L', 'U2R', 'normal']
    glob_cl = set(range(len(df.columns)))
    cl_idx = set([df.columns.get_loc(cl) for cl in classes])
    target_feature_idx = list(glob_cl.difference(cl_idx))
    cl_idx = list(cl_idx)
    dt = df.iloc[:, target_feature_idx]
    lb = df.iloc[:, cl_idx]
    assert len(dt) == len(lb), 'Something Wrong!!\nnumber of data is not equal to labels'
    if mode == 'np':
        return dt.to_numpy(), lb.to_numpy()
    elif mode == 'df':
        return dt, lb