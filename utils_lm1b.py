import numpy as np
import matplotlib.pyplot as plt

adv=np.load('test2.npz')
origin=np.load('test.npz')

print(origin)

datas=origin['datas']

origin_correct=origin['correct_data']
adv_correct=adv['correct_data']


print(origin_correct.sum()/datas.sum())
print(adv_correct.sum()/datas.sum())



adv_correct_ratio=adv_correct/datas
origin_correct_ratio=origin_correct/datas
print(adv_correct/datas)

print(origin_correct/datas)


diff=adv_correct_ratio-origin_correct_ratio

print(np.argsort(diff))


sorted_diff=np.sort(adv_correct_ratio-origin_correct_ratio)
print()


# plt.plot(np.arange(0,1000),adv_correct/datas)
# plt.plot(np.arange(0,1000),origin_correct/datas)
# plt.plot(np.arange(0,1000),sorted_diff)
# plt.plot(np.arange(0,1000),adv_correct_ratio-origin_correct_ratio)
# plt.plot(np.arange(0,1000),np.argsort(diff))
# plt.show()

print(sorted_diff)