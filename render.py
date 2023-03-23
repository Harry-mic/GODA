import gym
from mujoco_py import MjViewer, load_model_from_path, MjSim
import pdb
import os
import pandas as pd
import numpy as np
import time
import cv2
# 加载 HalfCheetah 模型
halfcheetah_xml = os.path.abspath("C:/1902why\software\code/research\GODA\GODA_MLP\data\halfcheetah.xml")
model = load_model_from_path(halfcheetah_xml)

# 创建 HalfCheetah 模拟器
sim = MjSim(model)
# 获取 HalfCheetah 状态
state = sim.get_state()
test_states = np.array(pd.read_csv("C:/1902why\software\code/research\GODA\GODA_MLP\data\states/3.22.18.48'.csv"))
viewer = MjViewer(sim)
viewer.cam.distance = 3
viewer.cam.elevation = -20
viewer.cam.lookat[0] = 0
viewer.cam.lookat[1] = 0
viewer.cam.lookat[2] = 0
for i in range(len(test_states)):
    state.qpos[:8] = test_states[i][:8]
    state.qvel[:9] = test_states[i][8:]
    sim.set_state(state)
    sim.forward()
    # viewer.render("C:/1902why\software\code/research\GODA\GODA_MLP\data\pic/image{}.png".format(i))
    image = sim.render(height=600, width=600, camera_name='track', depth=False)
    image = image[...,::-1]
    image = cv2.flip(image, 0)
    cv2.imwrite("C:/1902why\software\code/research\GODA\GODA_MLP\data\pic/3.22.18.48'{}.png".format(i), image)
