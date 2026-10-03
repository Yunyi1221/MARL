import copy
import sys
from datetime import datetime

import yaml

sys.path.append('D:\PythonProject\Multicell_UARA\module')
def load_env(env_path):
    '''
    :param model_path:
    # :param env_path:
    :return:
    '''
    with open(env_path, 'rb') as f:
        import pickle
        env = pickle.load(f)
    return env
class DotDic(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDic(copy.deepcopy(dict(self), memo=memo))

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
def system_shutdown(time_to_be_shutdown: int = 240):
    # import os
    # while True:
    #     time_to_be_shutdown -= 10
    #     print('Shutting down in {} sec'.format(time_to_be_shutdown))
    #     time.sleep(10)
    #     if time_to_be_shutdown <= 0:
    #         os.system(f"shutdown /s /t {1}")

    import tkinter as tk
    import os
    def shutdown():
        os.system(f"shutdown /s /t {time_to_be_shutdown}")
        print('shutdown!')

    def cancel_shutdown():
        os.system("shutdown -a")
        print('cancel_shutdown!')
        window.destroy()

    # 创建窗口
    window = tk.Tk()
    window.geometry("300x100")
    window.title("Cancel shutdown countdown?")
    shutdown()
    # 在窗口显示时开始倒计时
    button_cancel = tk.Button(window, text="Yes :D", command=cancel_shutdown, width=10, height=2)
    button_cancel.pack()
    button_cancel.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    # 运行窗口的主循环
    window.mainloop()

def get_time_str():
    now = datetime.now()
    return now.strftime("date%Y%m%dtime%H%M%S")

def get_config(path: str = None) -> DotDic:
    if not path:
        scepath = 'RConfig/config_environment_setting.yaml'
    else:
        scepath = path
    with open(scepath, 'r') as file:
        sce = DotDic(yaml.load(file, Loader=yaml.FullLoader))
    # Load the configuration file as arguments
    trial_sce = copy.deepcopy(sce)
    return trial_sce
import numpy as np
def check_constraint(z,a,Nb_,Nrb_):
    nBSs,nUEs,nRBs=a.shape
    z = np.array(z) if not isinstance(z,np.ndarray) else z
    a = np.array(a) if not isinstance(a,np.ndarray) else a
    for b in range(nBSs):
        if np.sum(z[b])!=Nb_:
            print(f'BS {b}: associates {np.sum(z[b])} UEs') #np.sum(z[b]) <= Nb  # 检查基站容量约束
        # assert np.sum(z[b]) <= Nb_
    rb_flag=0
    for u in range(nUEs):
        assert np.sum(z[:, u]) <= 1  # 检查用户唯一关联约束
        for b in range(nBSs):
            if z[b, u] == 1 and np.sum(a[b, u])!=Nrb_:
                print(f'UE {u}: occupy {np.sum(a[b, u])} RBs')
                rb_flag=1
                # assert np.sum(a[b, u]) <= Nrb_  # 检查用户RB分配约束
        if np.sum(np.sum(a[:,u,:],axis=-1) > 0) > 1:
            print(f'UEs {u} occupy RBGs from two BSs')
    if rb_flag==0:
        print(f'others UEs connect {Nrb_} RB')
