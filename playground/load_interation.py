import pickle as pkl

filename = "/home/yunbo/ScaDive/data/render_data/box_throw_ig_original/render_data.pkl"
interaction = pkl.load(open(filename,'rb'))
print(interaction)