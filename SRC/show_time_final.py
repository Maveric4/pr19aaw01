import codecs
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

IMG_NUMBER = 0

model_names = ['model_256x256', 'model_384x384', 'model_480x480']
file_path = './results_final.npy'
npy_obj = np.load(file_path, allow_pickle=True)
time_res = npy_obj[0][:, :, 1:]#[IMG_NUMBER, :, :]
test_iters = npy_obj[1][1:]

width = 0.5  # the width of the bars
labels = [str(x) for x in test_iters]
x = np.arange(len(labels))*1.5  # the label locations
# width_len = [-0.375, -0.125, 0.125, 0.375]  # dla 4 modeli
width_len = [-0.25, 0, 0.25]  # dla 3 modeli

fig, ax = plt.subplots()
plt.grid()
for i in range(0, len(model_names)):
    print(time_res[IMG_NUMBER, i, :])
    ax.bar(x + width_len[i], time_res[IMG_NUMBER, i, :], width/2, label=str(model_names[i]))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time [s]')
ax.set_xlabel('Iterations')
ax.set_title('Times by iteration and resolution')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

loc = plticker.MultipleLocator(base=0.025) # this locator puts ticks at regular intervals
ax.yaxis.set_major_locator(loc)
fig.tight_layout()
plt.legend(loc='upper right')

plt.savefig("./modele_256_384_480/show_time_final_fig.png")
plt.show()


## Figure for each model
width_len = [0]  # dla 1 modelu
for i in range(0, len(model_names)):
    fig, ax = plt.subplots()
    plt.grid()
    print(time_res[IMG_NUMBER, i, :])
    ax.bar(x + width_len, time_res[IMG_NUMBER, i, :], width/2, label=str(model_names[i]))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Iterations')
    ax.set_title('Times by iteration and resolution')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    loc = plticker.MultipleLocator(base=0.025) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    fig.tight_layout()
    print(i)
    plt.legend(loc='upper right')
    plt.savefig("./modele_256_384_480/show_time_final_fig_{}.png".format(i))
    plt.show()



